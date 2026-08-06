"""CCU V1 assembler / disassembler CLI and round-trip verifier."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

from .asm_text import format_program, parse_asm
from .codec import Instruction, decode_program, encode_program


def assemble_file(src: Path, dst: Path) -> int:
    text = src.read_text(encoding="utf-8")
    instrs = parse_asm(text)
    blob = encode_program(instrs)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(blob)
    return len(instrs)


def disassemble_file(src: Path, dst: Path, *, with_index: bool = False) -> int:
    blob = src.read_bytes()
    instrs = decode_program(blob)
    text = format_program(instrs, with_index=with_index)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")
    return len(instrs)


def compare_semantic(a: Sequence[Instruction], b: Sequence[Instruction]) -> List[str]:
    diffs: List[str] = []
    if len(a) != len(b):
        diffs.append(f"instruction count differs: source={len(a)} disasm={len(b)}")
    n = min(len(a), len(b))
    for i in range(n):
        ka = a[i].semantic_key()
        kb = b[i].semantic_key()
        if ka != kb:
            diffs.append(f"instr[{i}] mismatch:\n  source : {ka}\n  disasm : {kb}")
    for i in range(n, len(a)):
        diffs.append(f"instr[{i}] missing in disasm: {a[i].semantic_key()}")
    for i in range(n, len(b)):
        diffs.append(f"instr[{i}] extra in disasm: {b[i].semantic_key()}")
    return diffs


def verify_roundtrip(src: Path, work_dir: Path | None = None) -> Tuple[bool, str]:
    """Assemble -> disassemble -> compare semantic IR with source.

    Also reassembles the disassembly and checks binary equality with the first
    assembly output (reserved bits stay zero).
    """
    work = work_dir or (src.parent / ".ccu_v1_verify")
    work.mkdir(parents=True, exist_ok=True)
    bin_path = work / (src.stem + ".bin")
    dasm_path = work / (src.stem + ".dis.s")
    rebin_path = work / (src.stem + ".re.bin")

    src_instrs = parse_asm(src.read_text(encoding="utf-8"))
    assemble_file(src, bin_path)
    disassemble_file(bin_path, dasm_path)
    dasm_instrs = parse_asm(dasm_path.read_text(encoding="utf-8"))

    diffs = compare_semantic(src_instrs, dasm_instrs)
    blob1 = bin_path.read_bytes()
    assemble_file(dasm_path, rebin_path)
    blob2 = rebin_path.read_bytes()
    if blob1 != blob2:
        diffs.append(
            f"binary mismatch after re-assemble: {bin_path.name} ({len(blob1)}B) "
            f"vs {rebin_path.name} ({len(blob2)}B)"
        )

    if diffs:
        report = "FAIL semantic round-trip\n" + "\n".join(diffs)
        return False, report

    report = (
        f"OK: {len(src_instrs)} instructions\n"
        f"  source     : {src}\n"
        f"  binary     : {bin_path} ({len(blob1)} bytes)\n"
        f"  disasm     : {dasm_path}\n"
        f"  semantic   : source == disasm\n"
        f"  binary     : assemble(source) == assemble(disasm)\n"
    )
    return True, report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ccu_v1_asm",
        description="CCU V1 microcode assembler / disassembler",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("assemble", aliases=["as"], help="assemble .s -> .bin")
    a.add_argument("input", type=Path)
    a.add_argument("-o", "--output", type=Path, required=True)

    d = sub.add_parser("disassemble", aliases=["dis"], help="disassemble .bin -> .s")
    d.add_argument("input", type=Path)
    d.add_argument("-o", "--output", type=Path, required=True)
    d.add_argument("--index", action="store_true", help="emit instruction index comments")

    v = sub.add_parser(
        "verify",
        help="assemble, disassemble, compare semantic equivalence with source",
    )
    v.add_argument("input", type=Path, help="source assembly file")
    v.add_argument(
        "-w",
        "--work-dir",
        type=Path,
        default=None,
        help="directory for intermediate .bin / .dis.s (default: <src_dir>/.ccu_v1_verify)",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.cmd in ("assemble", "as"):
            n = assemble_file(args.input, args.output)
            print(f"assembled {n} instructions -> {args.output}")
            return 0
        if args.cmd in ("disassemble", "dis"):
            n = disassemble_file(args.input, args.output, with_index=args.index)
            print(f"disassembled {n} instructions -> {args.output}")
            return 0
        if args.cmd == "verify":
            ok, report = verify_roundtrip(args.input, args.work_dir)
            print(report)
            return 0 if ok else 1
    except Exception as e:  # noqa: BLE001 - CLI surface
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
