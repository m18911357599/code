#!/usr/bin/env python3
"""Tests for topology model, algorithm match, and collective simulation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import coll_op_dsl as cli  # noqa: E402
from program import parse_coll  # noqa: E402
from runtime import REDUCE_OPS, Trace, rec_dbl_allreduce, ring_allreduce  # noqa: E402
from topo_model import load_topo  # noqa: E402

EXAMPLES = ROOT / "examples"
RT_EX = ROOT.parent / "ranktable_dsl" / "examples"
FIXT = ROOT / "tests" / "fixtures"


class TestTopo(unittest.TestCase):
    def test_v2_two_layer(self) -> None:
        topo = load_topo(RT_EX / "v2_two_layer.rt")
        self.assertEqual(topo.rank_ids, [0, 1, 2, 3])
        self.assertEqual(topo.layer_ids(), [0, 1])
        l0 = topo.instances(0)
        self.assertEqual(len(l0), 2)
        self.assertEqual(l0[0].ranks, [0, 1])
        self.assertEqual(l0[1].ranks, [2, 3])
        self.assertEqual(topo.instances(1)[0].net_type, "CLOS")
        self.assertEqual(topo.instances(1)[0].ranks, [0, 1, 2, 3])
        self.assertEqual(topo.slot_groups(0), [[0, 2], [1, 3]])
        text = topo.describe()
        self.assertIn("az0-rack0-pod0", text)
        self.assertIn("CLOS", text)

    def test_v1_server(self) -> None:
        topo = load_topo(RT_EX / "v1_single_server.rt")
        self.assertEqual(len(topo.rank_ids), 8)
        self.assertEqual(topo.layer_ids(), [0])
        self.assertEqual(topo.instances(0)[0].ranks, list(range(8)))

    def test_v1_superpod(self) -> None:
        topo = load_topo(RT_EX / "v1_superpod.rt")
        self.assertEqual(topo.layer_ids(), [0, 1])
        self.assertEqual(len(topo.instances(0)), 2)
        self.assertEqual(topo.instances(1)[0].ranks, [0, 1, 2, 3])
        self.assertEqual(topo.slot_groups(0), [[0, 2], [1, 3]])


class TestRingPrimitives(unittest.TestCase):
    def test_ring_allreduce_n4(self) -> None:
        group = [0, 1, 2, 3]
        bufs = {r: [r * 4 + i for i in range(4)] for r in group}
        gold = [sum(bufs[r][i] for r in group) for i in range(4)]
        got = ring_allreduce(bufs, group, REDUCE_OPS["SUM"], Trace())
        for r in group:
            self.assertEqual(got[r], gold)

    def test_rec_dbl_matches_ring(self) -> None:
        group = [0, 1, 2, 3]
        bufs = {r: [r + 1] * 4 for r in group}
        a = ring_allreduce(bufs, group, REDUCE_OPS["SUM"], Trace())
        b = rec_dbl_allreduce(bufs, group, REDUCE_OPS["SUM"], Trace())
        for r in group:
            self.assertEqual(a[r], b[r])


class TestExamples(unittest.TestCase):
    def test_all_example_sims(self) -> None:
        for path in sorted(EXAMPLES.glob("*.coll")):
            with self.subTest(example=path.name):
                self.assertEqual(cli.main(["sim", str(path)]), 0, path.name)

    def test_ring_and_hier_same_result(self) -> None:
        prog_r, topo, _ = cli._load_prog_and_topo(str(EXAMPLES / "allreduce_ring.coll"))
        prog_h, _, _ = cli._load_prog_and_topo(str(EXAMPLES / "allreduce_hier.coll"))
        r = cli.verify_operator(prog_r, topo, prog_r.operators[0])
        h = cli.verify_operator(prog_h, topo, prog_h.operators[0])
        self.assertTrue(r["ok"])
        self.assertTrue(h["ok"])
        self.assertEqual(r["outputs"], h["outputs"])
        self.assertNotEqual(r["trace"].dump(), h["trace"].dump())
        self.assertNotEqual(r["steps"], h["steps"])

    def test_recursive_hd_rejected_on_three(self) -> None:
        src = """
        topo from "three_ranks.rt"
        operator AllReduce {
          reduce SUM
          count 4
          use recursive_hd on layer 0
        }
        """
        prog = parse_coll(src)
        topo = load_topo(FIXT / "three_ranks.rt")
        with self.assertRaises(cli.MatchError):
            cli.run_operator(prog, topo, prog.operators[0])

    def test_cli_topo(self) -> None:
        self.assertEqual(cli.main(["topo", str(RT_EX / "v2_two_layer.rt")]), 0)

    def test_cli_lower(self) -> None:
        self.assertEqual(cli.main(["lower", str(EXAMPLES / "allreduce_hier.coll"), "--rank", "0"]), 0)


if __name__ == "__main__":
    unittest.main()
