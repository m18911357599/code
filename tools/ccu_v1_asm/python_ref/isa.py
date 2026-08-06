"""CCU V1 ISA definitions: mnemonics, opcodes, and payload field layouts.

Binary layout matches hcomm CcuInstr (32 bytes, little-endian, #pragma pack(1)):
  header[u16]: code[10:0] | type[14:11] | reserved[15]
  payload[30B]: instruction-specific packed fields (bitfields LSB-first)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

INSTR_SIZE = 32
PAYLOAD_SIZE = 30

LOAD_TYPE = 0x0
CTRL_TYPE = 0x1
TRANS_TYPE = 0x2
REDUCE_TYPE = 0x3

# Field kinds used by the encoder/decoder.
# u16 / u64: fixed-width little-endian integers
# bits: packed into a shared u16 word; consecutive bit fields share one word
# ms_list: 8 x u16 Memory Slice ids
# flags_word: named bits packed into one u16 (explicit bit positions)


@dataclass(frozen=True)
class Field:
    name: str
    kind: str  # u16 | u64 | bits | ms_list | flags
    width: int = 16  # bits for bits/flags; unused for ms_list
    bit_offset: int = 0  # only for flags entries collected into one word


@dataclass(frozen=True)
class InstrDef:
    mnemonic: str
    type_: int
    code: int
    fields: Tuple[Field, ...]
    # Ordered operand names emitted by the disassembler (canonical form).
    operands: Tuple[str, ...]


def _u16(name: str) -> Field:
    return Field(name, "u16")


def _u64(name: str) -> Field:
    return Field(name, "u64", 64)


def _bits(name: str, width: int) -> Field:
    return Field(name, "bits", width)


def _ms_list(name: str = "ms") -> Field:
    return Field(name, "ms_list", 8 * 16)


def _rsv_u16(n: int) -> Tuple[Field, ...]:
    return tuple(Field(f"_rsv{i}", "u16") for i in range(n))


def _trans_tail_clear_len() -> Tuple[Field, ...]:
    """clearType:1 | lengthEn:1 | reserved:14, then set/wait CKE pair."""
    return (
        _bits("clear", 1),
        _bits("len_en", 1),
        _bits("_rsv_flags", 14),
        _u16("set_id"),
        _u16("set_mask"),
        _u16("wait_id"),
        _u16("wait_mask"),
    )


def _cke_tail() -> Tuple[Field, ...]:
    return (
        _u16("set_id"),
        _u16("set_mask"),
        _u16("wait_id"),
        _u16("wait_mask"),
    )


INSTRS: Tuple[InstrDef, ...] = (
    # ---- LOAD ----
    InstrDef(
        "LOAD_SQEARGS_TO_GSA",
        LOAD_TYPE,
        0x0,
        (_u16("gsa"), _u16("sqe")) + _rsv_u16(13),
        ("gsa", "sqe"),
    ),
    InstrDef(
        "LOAD_SQEARGS_TO_XN",
        LOAD_TYPE,
        0x1,
        (_u16("xn"), _u16("sqe")) + _rsv_u16(13),
        ("xn", "sqe"),
    ),
    InstrDef(
        "LOAD_IMD_TO_GSA",
        LOAD_TYPE,
        0x2,
        (_u16("gsa"), _u64("imm")) + _rsv_u16(10),
        ("gsa", "imm"),
    ),
    InstrDef(
        "LOAD_IMD_TO_XN",
        LOAD_TYPE,
        0x3,
        (_u16("xn"), _u64("imm"), _u16("sec")) + _rsv_u16(9),
        ("xn", "imm", "sec"),
    ),
    InstrDef(
        "LOAD_GSA_XN",
        LOAD_TYPE,
        0x4,
        (_u16("gsad"), _u16("gsam"), _u16("xn")) + _rsv_u16(12),
        ("gsad", "gsam", "xn"),
    ),
    InstrDef(
        "LOAD_GSA_GSA",
        LOAD_TYPE,
        0x5,
        (_u16("gsad"), _u16("gsam"), _u16("gsan")) + _rsv_u16(12),
        ("gsad", "gsam", "gsan"),
    ),
    InstrDef(
        "LOAD_XX",
        LOAD_TYPE,
        0x6,
        (_u16("xd"), _u16("xm"), _u16("xn")) + _rsv_u16(12),
        ("xd", "xm", "xn"),
    ),
    # ---- CTRL ----
    InstrDef(
        "LOOP",
        CTRL_TYPE,
        0x0,
        (_u16("start"), _u16("end"), _u16("xn")) + _rsv_u16(12),
        ("start", "end", "xn"),
    ),
    InstrDef(
        "LOOP_GROUP",
        CTRL_TYPE,
        0x1,
        (_u16("start_loop"), _u16("xn"), _u16("xm"), _bits("hiperf", 1), _bits("_rsv_hp", 15))
        + _rsv_u16(11),
        ("start_loop", "xn", "xm", "hiperf"),
    ),
    InstrDef(
        "SET_CKE",
        CTRL_TYPE,
        0x2,
        (_bits("clear", 1), _bits("_rsv_c", 15), _u16("set_id"), _u16("set_mask"), _u16("wait_id"), _u16("wait_mask"))
        + _rsv_u16(10),
        ("clear", "set_id", "set_mask", "wait_id", "wait_mask"),
    ),
    InstrDef(
        "CLEAR_CKE",
        CTRL_TYPE,
        0x4,
        (
            _bits("clear", 1),
            _bits("_rsv_c", 15),
            _u16("clear_id"),
            _u16("clear_mask"),
            _u16("wait_id"),
            _u16("wait_mask"),
        )
        + _rsv_u16(10),
        ("clear", "clear_id", "clear_mask", "wait_id", "wait_mask"),
    ),
    InstrDef(
        "JMP",
        CTRL_TYPE,
        0x5,
        # Only 12 bytes of meaningful payload; remaining 18 bytes zero-filled.
        (_u16("dst_xn"), _u16("cond_xn"), _u64("expect")) + _rsv_u16(9),
        ("dst_xn", "cond_xn", "expect"),
    ),
    # ---- TRANS mem/ms ----
    InstrDef(
        "TRANS_LOC_MEM_TO_LOC_MS",
        TRANS_TYPE,
        0x0,
        (_u16("ms"), _u16("gsa"), _u16("xn"), _u16("len_xn"), _u16("ch"))
        + _rsv_u16(5)
        + _trans_tail_clear_len(),
        ("ms", "gsa", "xn", "len_xn", "ch", "clear", "len_en", "set_id", "set_mask", "wait_id", "wait_mask"),
    ),
    InstrDef(
        "TRANS_RMT_MEM_TO_LOC_MS",
        TRANS_TYPE,
        0x1,
        (_u16("ms"), _u16("gsa"), _u16("xn"), _u16("len_xn"), _u16("ch"))
        + _rsv_u16(5)
        + _trans_tail_clear_len(),
        ("ms", "gsa", "xn", "len_xn", "ch", "clear", "len_en", "set_id", "set_mask", "wait_id", "wait_mask"),
    ),
    InstrDef(
        "TRANS_LOC_MS_TO_LOC_MEM",
        TRANS_TYPE,
        0x2,
        (_u16("gsa"), _u16("xn"), _u16("ms"), _u16("len_xn"), _u16("ch"))
        + _rsv_u16(5)
        + _trans_tail_clear_len(),
        ("gsa", "xn", "ms", "len_xn", "ch", "clear", "len_en", "set_id", "set_mask", "wait_id", "wait_mask"),
    ),
    InstrDef(
        "TRANS_LOC_MS_TO_RMT_MEM",
        TRANS_TYPE,
        0x3,
        (_u16("gsa"), _u16("xn"), _u16("ms"), _u16("len_xn"), _u16("ch"))
        + _rsv_u16(5)
        + _trans_tail_clear_len(),
        ("gsa", "xn", "ms", "len_xn", "ch", "clear", "len_en", "set_id", "set_mask", "wait_id", "wait_mask"),
    ),
    InstrDef(
        "TRANS_RMT_MS_TO_LOC_MEM",
        TRANS_TYPE,
        0x4,
        (_u16("gsa"), _u16("xn"), _u16("ms"), _u16("len_xn"), _u16("ch"))
        + _rsv_u16(5)
        + _trans_tail_clear_len(),
        ("gsa", "xn", "ms", "len_xn", "ch", "clear", "len_en", "set_id", "set_mask", "wait_id", "wait_mask"),
    ),
    InstrDef(
        "TRANS_LOC_MS_TO_LOC_MS",
        TRANS_TYPE,
        0x5,
        (_u16("dst_ms"), _u16("src_ms"), _u16("len_xn"), _u16("ch"))
        + _rsv_u16(6)
        + _trans_tail_clear_len(),
        ("dst_ms", "src_ms", "len_xn", "ch", "clear", "len_en", "set_id", "set_mask", "wait_id", "wait_mask"),
    ),
    InstrDef(
        "TRANS_RMT_MS_TO_LOC_MS",
        TRANS_TYPE,
        0x6,
        (_u16("loc_ms"), _u16("rmt_ms"), _u16("len_xn"), _u16("ch"))
        + _rsv_u16(6)
        + _trans_tail_clear_len(),
        ("loc_ms", "rmt_ms", "len_xn", "ch", "clear", "len_en", "set_id", "set_mask", "wait_id", "wait_mask"),
    ),
    InstrDef(
        "TRANS_LOC_MS_TO_RMT_MS",
        TRANS_TYPE,
        0x7,
        (
            _u16("rmt_ms"),
            _u16("loc_ms"),
            _u16("len_xn"),
            _u16("ch"),
            _u16("rmt_set_id"),
            _u16("rmt_set_mask"),
        )
        + _rsv_u16(4)
        + _trans_tail_clear_len(),
        (
            "rmt_ms",
            "loc_ms",
            "len_xn",
            "ch",
            "rmt_set_id",
            "rmt_set_mask",
            "clear",
            "len_en",
            "set_id",
            "set_mask",
            "wait_id",
            "wait_mask",
        ),
    ),
    InstrDef(
        "TRANS_RMT_MEM_TO_LOC_MEM",
        TRANS_TYPE,
        0x8,
        (
            _u16("loc_gsa"),
            _u16("loc_xn"),
            _u16("rmt_gsa"),
            _u16("rmt_xn"),
            _u16("len_xn"),
            _u16("ch"),
            _bits("udf", 8),
            _bits("reduce_dtype", 4),
            _bits("reduce_op", 4),
        )
        + _rsv_u16(3)
        + (
            _bits("clear", 1),
            _bits("len_en", 1),
            _bits("reduce_en", 1),
            _bits("_rsv_f", 13),
        )
        + _cke_tail(),
        (
            "loc_gsa",
            "loc_xn",
            "rmt_gsa",
            "rmt_xn",
            "len_xn",
            "ch",
            "udf",
            "reduce_dtype",
            "reduce_op",
            "clear",
            "len_en",
            "reduce_en",
            "set_id",
            "set_mask",
            "wait_id",
            "wait_mask",
        ),
    ),
    InstrDef(
        "TRANS_LOC_MEM_TO_RMT_MEM",
        TRANS_TYPE,
        0x9,
        (
            _u16("rmt_gsa"),
            _u16("rmt_xn"),
            _u16("loc_gsa"),
            _u16("loc_xn"),
            _u16("len_xn"),
            _u16("ch"),
            _bits("udf", 8),
            _bits("reduce_dtype", 4),
            _bits("reduce_op", 4),
        )
        + _rsv_u16(3)
        + (
            _bits("clear", 1),
            _bits("len_en", 1),
            _bits("reduce_en", 1),
            _bits("_rsv_f", 13),
        )
        + _cke_tail(),
        (
            "rmt_gsa",
            "rmt_xn",
            "loc_gsa",
            "loc_xn",
            "len_xn",
            "ch",
            "udf",
            "reduce_dtype",
            "reduce_op",
            "clear",
            "len_en",
            "reduce_en",
            "set_id",
            "set_mask",
            "wait_id",
            "wait_mask",
        ),
    ),
    InstrDef(
        "TRANS_LOC_MEM_TO_LOC_MEM",
        TRANS_TYPE,
        0xA,
        (_u16("dst_gsa"), _u16("dst_xn"), _u16("src_gsa"), _u16("src_xn"), _u16("len_xn"), _u16("ch"))
        + _rsv_u16(4)
        + _trans_tail_clear_len(),
        (
            "dst_gsa",
            "dst_xn",
            "src_gsa",
            "src_xn",
            "len_xn",
            "ch",
            "clear",
            "len_en",
            "set_id",
            "set_mask",
            "wait_id",
            "wait_mask",
        ),
    ),
    # ---- TRANS sync ----
    InstrDef(
        "SYNC_CKE",
        TRANS_TYPE,
        0xB,
        (_u16("rmt_cke"), _u16("loc_cke"), _u16("loc_mask"), _u16("ch"))
        + _rsv_u16(6)
        + (_bits("clear", 1), _bits("_rsv_c", 15))
        + _cke_tail(),
        ("rmt_cke", "loc_cke", "loc_mask", "ch", "clear", "set_id", "set_mask", "wait_id", "wait_mask"),
    ),
    InstrDef(
        "SYNC_GSA",
        TRANS_TYPE,
        0xC,
        (
            _u16("rmt_gsa"),
            _u16("loc_gsa"),
            _u16("_rsv2"),
            _u16("ch"),
            _u16("rmt_set_id"),
            _u16("rmt_set_mask"),
        )
        + _rsv_u16(4)
        + (_bits("clear", 1), _bits("_rsv_c", 15))
        + _cke_tail(),
        (
            "rmt_gsa",
            "loc_gsa",
            "ch",
            "rmt_set_id",
            "rmt_set_mask",
            "clear",
            "set_id",
            "set_mask",
            "wait_id",
            "wait_mask",
        ),
    ),
    InstrDef(
        "SYNC_XN",
        TRANS_TYPE,
        0xD,
        (
            _u16("rmt_xn"),
            _u16("loc_xn"),
            _u16("_rsv2"),
            _u16("ch"),
            _u16("rmt_set_id"),
            _u16("rmt_set_mask"),
        )
        + _rsv_u16(4)
        + (_bits("clear", 1), _bits("_rsv_c", 15))
        + _cke_tail(),
        (
            "rmt_xn",
            "loc_xn",
            "ch",
            "rmt_set_id",
            "rmt_set_mask",
            "clear",
            "set_id",
            "set_mask",
            "wait_id",
            "wait_mask",
        ),
    ),
    # ---- REDUCE ----
    InstrDef(
        "ADD",
        REDUCE_TYPE,
        0x0,
        (
            _ms_list("ms"),
            _u16("len_xn"),
            _u16("_rsv"),
            _bits("clear", 1),
            _bits("count", 3),
            _bits("cast", 2),
            _bits("_rsv1", 5),
            _bits("dtype", 5),
        )
        + _cke_tail(),
        ("ms", "count", "cast", "dtype", "len_xn", "clear", "set_id", "set_mask", "wait_id", "wait_mask"),
    ),
    InstrDef(
        "MAX",
        REDUCE_TYPE,
        0x1,
        (
            _ms_list("ms"),
            _u16("len_xn"),
            _u16("_rsv"),
            _bits("clear", 1),
            _bits("count", 3),
            _bits("_rsv1", 7),
            _bits("dtype", 5),
        )
        + _cke_tail(),
        ("ms", "count", "dtype", "len_xn", "clear", "set_id", "set_mask", "wait_id", "wait_mask"),
    ),
    InstrDef(
        "MIN",
        REDUCE_TYPE,
        0x2,
        (
            _ms_list("ms"),
            _u16("len_xn"),
            _u16("_rsv"),
            _bits("clear", 1),
            _bits("count", 3),
            _bits("_rsv1", 7),
            _bits("dtype", 5),
        )
        + _cke_tail(),
        ("ms", "count", "dtype", "len_xn", "clear", "set_id", "set_mask", "wait_id", "wait_mask"),
    ),
)

BY_MNEMONIC: Dict[str, InstrDef] = {i.mnemonic: i for i in INSTRS}
BY_OPCODE: Dict[Tuple[int, int], InstrDef] = {(i.type_, i.code): i for i in INSTRS}


def make_header(type_: int, code: int) -> int:
    if not (0 <= type_ < 16 and 0 <= code < 2048):
        raise ValueError(f"invalid type/code: type={type_} code={code}")
    return (type_ << 11) | code


def parse_header(raw: int) -> Tuple[int, int]:
    code = raw & 0x7FF
    type_ = (raw >> 11) & 0xF
    return type_, code


def payload_bit_length(fields: Sequence[Field]) -> int:
    total = 0
    for f in fields:
        if f.kind == "ms_list":
            total += 8 * 16
        else:
            total += f.width if f.kind in ("bits", "u64") else 16
    return total


# Sanity: every instruction payload must be exactly 30 bytes (240 bits).
for _idef in INSTRS:
    _bits_len = payload_bit_length(_idef.fields)
    if _bits_len != PAYLOAD_SIZE * 8:
        raise RuntimeError(
            f"payload size mismatch for {_idef.mnemonic}: {_bits_len} bits "
            f"(expected {PAYLOAD_SIZE * 8})"
        )
