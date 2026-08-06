#!/usr/bin/env python3
"""Unit / integration tests for CCU V1 assembler & disassembler."""

from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccu_v1_asm.asm_text import format_instruction, format_program, parse_asm
from ccu_v1_asm.cli import verify_roundtrip
from ccu_v1_asm.codec import decode_instruction, decode_program, encode_instruction, encode_program
from ccu_v1_asm.isa import BY_MNEMONIC, INSTRS, INSTR_SIZE, make_header

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


class TestHeader(unittest.TestCase):
    def test_header_matches_c_layout(self):
        # type=2, code=0xa -> 0x100a (from GCC layout check)
        self.assertEqual(make_header(2, 0xA), 0x100A)
        self.assertEqual(make_header(0, 2), 0x0002)


class TestPayloadLayouts(unittest.TestCase):
    def test_load_imd_to_gsa_bytes(self):
        src = "LOAD_IMD_TO_GSA gsa=7, imm=0x1122334455667788"
        instr = parse_asm(src)[0]
        blob = encode_instruction(instr)
        self.assertEqual(len(blob), INSTR_SIZE)
        # Matches /tmp/check_layout.c dump
        expected = bytes(
            [
                0x02,
                0x00,
                0x07,
                0x00,
                0x88,
                0x77,
                0x66,
                0x55,
                0x44,
                0x33,
                0x22,
                0x11,
            ]
            + [0x00] * 20
        )
        self.assertEqual(blob, expected)

    def test_trans_flags_word(self):
        src = (
            "TRANS_LOC_MEM_TO_LOC_MS ms=0, gsa=0, xn=0, len_xn=0, ch=0, "
            "clear=1, len_en=1, set_id=0, set_mask=0, wait_id=0, wait_mask=0"
        )
        blob = encode_instruction(parse_asm(src)[0])
        flags = struct.unpack_from("<H", blob, 2 + 20)[0]  # after 5 u16 + 5 reserved
        self.assertEqual(flags, 0x0003)

    def test_add_flags_word(self):
        src = (
            "ADD ms=[0,0,0,0,0,0,0,0], count=7, cast=3, dtype=31, len_xn=0, "
            "clear=1, set_id=0, set_mask=0, wait_id=0, wait_mask=0"
        )
        blob = encode_instruction(parse_asm(src)[0])
        flags = struct.unpack_from("<H", blob, 2 + 16 + 2 + 2)[0]
        self.assertEqual(flags, 0xF83F)

    def test_mem2mem_reduce_word(self):
        src = (
            "TRANS_LOC_MEM_TO_RMT_MEM rmt_gsa=0, rmt_xn=0, loc_gsa=0, loc_xn=0, "
            "len_xn=0, ch=0, udf=0xab, reduce_dtype=0xc, reduce_op=0x5, "
            "clear=1, len_en=1, reduce_en=1, set_id=0, set_mask=0, wait_id=0, wait_mask=0"
        )
        blob = encode_instruction(parse_asm(src)[0])
        reduce_word = struct.unpack_from("<H", blob, 2 + 12)[0]
        flags = struct.unpack_from("<H", blob, 2 + 12 + 2 + 6)[0]
        self.assertEqual(reduce_word, 0x5CAB)
        self.assertEqual(flags, 0x0007)


class TestRoundTrip(unittest.TestCase):
    def test_each_opcode_alone(self):
        from ccu_v1_asm.codec import Instruction

        for idef in INSTRS:
            ops = {}
            for name in idef.operands:
                if name == "ms" and any(f.name == "ms" and f.kind == "ms_list" for f in idef.fields):
                    ops[name] = [0] * 8
                elif name == "count":
                    ops[name] = 2
                else:
                    ops[name] = 0
            instr = Instruction(idef.mnemonic, ops).normalized()
            blob = encode_instruction(instr)
            back = decode_instruction(blob)
            self.assertEqual(instr.semantic_key(), back.semantic_key(), idef.mnemonic)
            # text round-trip
            text = format_instruction(instr)
            back2 = parse_asm(text)[0]
            self.assertEqual(instr.semantic_key(), back2.semantic_key(), text)

    def test_all_opcodes_example_file(self):
        src = EXAMPLES / "all_opcodes.s"
        with tempfile.TemporaryDirectory() as td:
            ok, report = verify_roundtrip(src, Path(td))
            self.assertTrue(ok, report)

    def test_operand_order_independent(self):
        a = parse_asm("LOAD_XX xn=6, xd=4, xm=5")[0]
        b = parse_asm("LOAD_XX xd=4, xm=5, xn=6")[0]
        self.assertEqual(a.semantic_key(), b.semantic_key())
        self.assertEqual(encode_instruction(a), encode_instruction(b))

    def test_disasm_text_reassembles_identical(self):
        src_text = (EXAMPLES / "all_opcodes.s").read_text(encoding="utf-8")
        instrs = parse_asm(src_text)
        blob = encode_program(instrs)
        dasm = format_program(decode_program(blob))
        blob2 = encode_program(parse_asm(dasm))
        self.assertEqual(blob, blob2)

    def test_source_vs_disasm_semantic(self):
        """User requirement: disasm output vs source — semantic equivalence."""
        src = EXAMPLES / "all_opcodes.s"
        src_instrs = parse_asm(src.read_text(encoding="utf-8"))
        blob = encode_program(src_instrs)
        dasm_instrs = decode_program(blob)
        self.assertEqual(len(src_instrs), len(dasm_instrs))
        for i, (a, b) in enumerate(zip(src_instrs, dasm_instrs)):
            self.assertEqual(a.semantic_key(), b.semantic_key(), f"instr {i}")


class TestCliVerify(unittest.TestCase):
    def test_verify_cli_module(self):
        ok, report = verify_roundtrip(EXAMPLES / "all_opcodes.s")
        self.assertTrue(ok, report)


if __name__ == "__main__":
    unittest.main()
