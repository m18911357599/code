"""Encode / decode CCU V1 instructions between semantic IR and 32-byte binary."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from .isa import (
    BY_MNEMONIC,
    BY_OPCODE,
    INSTR_SIZE,
    PAYLOAD_SIZE,
    Field,
    InstrDef,
    make_header,
    parse_header,
)


@dataclass
class Instruction:
    """Semantic IR for one CCU V1 instruction."""

    mnemonic: str
    operands: Dict[str, Any]  # ints, or List[int] for ms

    def normalized(self) -> "Instruction":
        idef = BY_MNEMONIC[self.mnemonic]
        list_fields = {f.name for f in idef.fields if f.kind == "ms_list"}
        ops: Dict[str, Any] = {}
        for name in idef.operands:
            val = self.operands.get(name, 0)
            if name in list_fields:
                ms = list(val) if isinstance(val, (list, tuple)) else [int(val)]
                while len(ms) < 8:
                    ms.append(0)
                ops[name] = [int(x) & 0xFFFF for x in ms[:8]]
            else:
                ops[name] = int(val)
        return Instruction(self.mnemonic, ops)

    def semantic_key(self) -> Tuple:
        n = self.normalized()
        items = []
        for k in BY_MNEMONIC[n.mnemonic].operands:
            v = n.operands[k]
            if isinstance(v, list):
                items.append((k, tuple(v)))
            else:
                items.append((k, v))
        return (n.mnemonic, tuple(items))


def _pack_bitfields(fields: Sequence[Field], values: Mapping[str, Any], start: int) -> Tuple[bytes, int]:
    """Pack consecutive 'bits' fields into one or more little-endian u16 words."""
    out = bytearray()
    i = start
    while i < len(fields) and fields[i].kind == "bits":
        word = 0
        bitpos = 0
        while i < len(fields) and fields[i].kind == "bits" and bitpos + fields[i].width <= 16:
            f = fields[i]
            raw = 0 if f.name.startswith("_") else int(values.get(f.name, 0))
            mask = (1 << f.width) - 1
            if raw & ~mask:
                raise ValueError(f"field {f.name}={raw} does not fit in {f.width} bits")
            word |= (raw & mask) << bitpos
            bitpos += f.width
            i += 1
            if bitpos == 16:
                break
        if bitpos != 16:
            # Should not happen if ISA defs pack to whole words.
            raise ValueError(f"bitfield group not aligned to 16 bits (got {bitpos})")
        out += struct.pack("<H", word)
    return bytes(out), i


def encode_payload(idef: InstrDef, operands: Mapping[str, Any]) -> bytes:
    buf = bytearray()
    i = 0
    fields = idef.fields
    while i < len(fields):
        f = fields[i]
        if f.kind == "u16":
            val = 0 if f.name.startswith("_") else int(operands.get(f.name, 0))
            if not 0 <= val <= 0xFFFF:
                raise ValueError(f"{idef.mnemonic}.{f.name}={val} out of u16 range")
            buf += struct.pack("<H", val)
            i += 1
        elif f.kind == "u64":
            val = int(operands.get(f.name, 0))
            if not 0 <= val <= 0xFFFFFFFFFFFFFFFF:
                raise ValueError(f"{idef.mnemonic}.{f.name}={val} out of u64 range")
            buf += struct.pack("<Q", val)
            i += 1
        elif f.kind == "ms_list":
            ms = operands.get(f.name, [0] * 8)
            if not isinstance(ms, (list, tuple)):
                raise ValueError(f"{idef.mnemonic}.ms must be a list")
            ms_list = list(ms)
            while len(ms_list) < 8:
                ms_list.append(0)
            for x in ms_list[:8]:
                xi = int(x)
                if not 0 <= xi <= 0xFFFF:
                    raise ValueError(f"ms id {xi} out of u16 range")
                buf += struct.pack("<H", xi)
            i += 1
        elif f.kind == "bits":
            chunk, i = _pack_bitfields(fields, operands, i)
            buf += chunk
        else:
            raise ValueError(f"unknown field kind {f.kind}")
    if len(buf) != PAYLOAD_SIZE:
        raise RuntimeError(f"{idef.mnemonic}: encoded payload {len(buf)} != {PAYLOAD_SIZE}")
    return bytes(buf)


def decode_payload(idef: InstrDef, payload: bytes) -> Dict[str, Any]:
    if len(payload) != PAYLOAD_SIZE:
        raise ValueError(f"payload must be {PAYLOAD_SIZE} bytes")
    ops: Dict[str, Any] = {}
    off = 0
    i = 0
    fields = idef.fields
    while i < len(fields):
        f = fields[i]
        if f.kind == "u16":
            (val,) = struct.unpack_from("<H", payload, off)
            off += 2
            if not f.name.startswith("_"):
                ops[f.name] = val
            i += 1
        elif f.kind == "u64":
            (val,) = struct.unpack_from("<Q", payload, off)
            off += 8
            ops[f.name] = val
            i += 1
        elif f.kind == "ms_list":
            ms = list(struct.unpack_from("<8H", payload, off))
            off += 16
            ops[f.name] = ms
            i += 1
        elif f.kind == "bits":
            (word,) = struct.unpack_from("<H", payload, off)
            off += 2
            bitpos = 0
            while i < len(fields) and fields[i].kind == "bits" and bitpos + fields[i].width <= 16:
                bf = fields[i]
                mask = (1 << bf.width) - 1
                val = (word >> bitpos) & mask
                if not bf.name.startswith("_"):
                    ops[bf.name] = val
                bitpos += bf.width
                i += 1
                if bitpos == 16:
                    break
            if bitpos != 16:
                raise ValueError(f"bitfield group not 16-bit aligned for {idef.mnemonic}")
        else:
            raise ValueError(f"unknown field kind {f.kind}")
    return ops


def encode_instruction(instr: Instruction) -> bytes:
    if instr.mnemonic not in BY_MNEMONIC:
        raise ValueError(f"unknown mnemonic: {instr.mnemonic}")
    idef = BY_MNEMONIC[instr.mnemonic]
    norm = instr.normalized()
    header = make_header(idef.type_, idef.code)
    payload = encode_payload(idef, norm.operands)
    return struct.pack("<H", header) + payload


def decode_instruction(data: bytes) -> Instruction:
    if len(data) != INSTR_SIZE:
        raise ValueError(f"instruction must be {INSTR_SIZE} bytes, got {len(data)}")
    (header,) = struct.unpack_from("<H", data, 0)
    type_, code = parse_header(header)
    key = (type_, code)
    if key not in BY_OPCODE:
        raise ValueError(f"unknown opcode type=0x{type_:x} code=0x{code:x} (header=0x{header:04x})")
    idef = BY_OPCODE[key]
    ops = decode_payload(idef, data[2:])
    return Instruction(idef.mnemonic, ops).normalized()


def encode_program(instrs: Sequence[Instruction]) -> bytes:
    return b"".join(encode_instruction(i) for i in instrs)


def decode_program(data: bytes) -> List[Instruction]:
    if len(data) % INSTR_SIZE != 0:
        raise ValueError(f"binary size {len(data)} is not a multiple of {INSTR_SIZE}")
    out: List[Instruction] = []
    for off in range(0, len(data), INSTR_SIZE):
        out.append(decode_instruction(data[off : off + INSTR_SIZE]))
    return out
