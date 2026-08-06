"""Assembly text parser and formatter for CCU V1."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

from codec import Instruction
from isa import BY_MNEMONIC

_LINE_RE = re.compile(
    r"^\s*(?:(?P<label>[A-Za-z_][\w]*)\s*:)?\s*"
    r"(?:(?P<mnem>[A-Za-z_][\w]*)\s*(?P<ops>.*?)?)?\s*(?:#.*)?$"
)
_OP_RE = re.compile(
    r"(?P<name>[A-Za-z_][\w]*)\s*=\s*"
    r"(?P<val>\[[^\]]*\]|0[xX][0-9A-Fa-f]+|\d+)"
)


def parse_int(text: str) -> int:
    text = text.strip()
    if text.lower().startswith("0x"):
        return int(text, 16)
    return int(text, 10)


def parse_ms_list(text: str) -> List[int]:
    inner = text.strip()
    if not (inner.startswith("[") and inner.endswith("]")):
        raise ValueError(f"ms list expected like [0,1,2], got {text!r}")
    body = inner[1:-1].strip()
    if not body:
        return []
    return [parse_int(p) for p in body.split(",")]


def parse_operands(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        return {}
    ops: Dict[str, Any] = {}
    pos = 0
    for m in _OP_RE.finditer(text):
        # ensure only commas/spaces between operands
        gap = text[pos : m.start()]
        if gap.strip() not in ("", ","):
            raise ValueError(f"unexpected operand text near {gap!r}")
        name = m.group("name")
        raw = m.group("val")
        if raw.startswith("["):
            ops[name] = parse_ms_list(raw)
        else:
            ops[name] = parse_int(raw)
        pos = m.end()
    trailing = text[pos:].strip().strip(",")
    if trailing:
        raise ValueError(f"unparsed operand tail: {trailing!r}")
    return ops


def parse_asm(text: str) -> List[Instruction]:
    """Parse assembly source into Instruction IR.

    Supports:
      - blank lines and `#` comments
      - optional labels: `L0: MNEMONIC a=1, b=2`
      - named operands only (order-independent on input; canonical on output)
    """
    instrs: List[Instruction] = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = _LINE_RE.match(raw)
        if not m:
            raise ValueError(f"line {lineno}: syntax error: {raw}")
        mnem = m.group("mnem")
        if not mnem:
            # label-only line
            continue
        mnem = mnem.upper()
        if mnem not in BY_MNEMONIC:
            raise ValueError(f"line {lineno}: unknown mnemonic {mnem}")
        try:
            ops = parse_operands(m.group("ops") or "")
        except ValueError as e:
            raise ValueError(f"line {lineno}: {e}") from e
        # Reject unknown operand names early.
        allowed = set(BY_MNEMONIC[mnem].operands)
        unknown = set(ops) - allowed
        if unknown:
            raise ValueError(f"line {lineno}: unknown operands for {mnem}: {sorted(unknown)}")
        instrs.append(Instruction(mnem, ops).normalized())
    return instrs


def format_value(name: str, value: Any) -> str:
    if isinstance(value, list):
        return "[" + ",".join(str(int(x)) for x in value) + "]"
    iv = int(value)
    if name in ("imm", "expect", "set_mask", "wait_mask", "clear_mask", "loc_mask", "rmt_set_mask"):
        return f"0x{iv:x}"
    return str(iv)


def format_instruction(instr: Instruction) -> str:
    n = instr.normalized()
    idef = BY_MNEMONIC[n.mnemonic]
    parts = [f"{name}={format_value(name, n.operands[name])}" for name in idef.operands]
    return n.mnemonic + (" " + ", ".join(parts) if parts else "")


def format_program(instrs: Sequence[Instruction], *, with_index: bool = False) -> str:
    lines: List[str] = []
    for i, instr in enumerate(instrs):
        line = format_instruction(instr)
        if with_index:
            line = f"# [{i}]\n{line}"
        lines.append(line)
    return "\n".join(lines) + ("\n" if lines else "")
