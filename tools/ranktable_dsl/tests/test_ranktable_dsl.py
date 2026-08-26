#!/usr/bin/env python3
"""Round-trip and constraint tests for the rank table DSL."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ranktable_dsl as rt  # noqa: E402

EXAMPLES = ROOT / "examples"
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _ir(table: rt.RankTable) -> dict:
    return rt.emit_json(table)


class TestHcommFixtures(unittest.TestCase):
    def test_decompile_v2_fixture(self) -> None:
        data = json.loads((FIXTURES / "rank_table_v2.json").read_text())
        table = rt.load_json(data)
        rt.check(table)
        self.assertEqual(table.version, "2.0")
        self.assertEqual(table.rank_count(), 2)
        self.assertEqual(table.ranks[0].levels[0].net_instance_id, "az0-rack0")
        self.assertEqual(table.ranks[0].levels[0].addrs[0].ports, ["0/0"])

        again = rt.parse_dsl(rt.emit_dsl(table))
        rt.check(again)
        self.assertEqual(_ir(table), _ir(again))

    def test_decompile_v1_fixture(self) -> None:
        data = json.loads((FIXTURES / "rank_table_v1.json").read_text())
        table = rt.load_json(data)
        rt.check(table)
        self.assertEqual(table.version, "1.0")
        self.assertEqual(table.rank_count(), 8)
        self.assertEqual(table.servers[0].server_id, "SERVER_ID_SV1")
        self.assertEqual(table.servers[0].devices[-1].device_ip, "192.168.1.15")

        again = rt.parse_dsl(rt.emit_dsl(table))
        self.assertEqual(_ir(table), _ir(again))


class TestExamples(unittest.TestCase):
    def test_compile_all_examples(self) -> None:
        for path in sorted(EXAMPLES.glob("*.rt")):
            with self.subTest(example=path.name):
                table = rt.parse_dsl(path.read_text())
                rt.check(table)
                dumped = rt.emit_dsl(table)
                again = rt.parse_dsl(dumped)
                self.assertEqual(_ir(table), _ir(again), path.name)

    def test_v2_two_layer_fields(self) -> None:
        table = rt.parse_dsl((EXAMPLES / "v2_two_layer.rt").read_text())
        rt.check(table)
        self.assertEqual(len(table.ranks), 4)
        r0 = table.ranks[0]
        self.assertEqual(r0.device_port, 16666)
        self.assertEqual(r0.levels[0].net_type, "TOPO_FILE_DESC")
        self.assertEqual(r0.levels[1].net_type, "CLOS")
        self.assertEqual(r0.levels[1].addrs[0].plane_id, "plane0")
        self.assertEqual(r0.levels[1].addrs[1].backup_addrs, ["172.16.0.6"])
        self.assertIsNotNone(r0.control_plane)
        assert r0.control_plane is not None
        self.assertEqual(r0.control_plane.listen_port, 16666)
        js = rt.emit_json(table)
        self.assertEqual(js["rank_count"], 4)
        self.assertEqual(js["rank_list"][0]["control_plane"]["listen_port"], 16666)

    def test_v1_superpod(self) -> None:
        table = rt.parse_dsl((EXAMPLES / "v1_superpod.rt").read_text())
        rt.check(table)
        js = rt.emit_json(table)
        self.assertEqual(js["version"], "1.2")
        self.assertEqual(js["server_count"], "2")
        self.assertEqual(js["server_list"][0]["host_ip"], "172.16.0.100")
        self.assertEqual(js["server_list"][0]["device"][0]["super_device_id"], "0")
        self.assertEqual(js["super_pod_list"][0]["server_list"][1]["server_id"], "node_1")


class TestChecks(unittest.TestCase):
    def test_v2_version_must_be_20(self) -> None:
        src = """
        ranktable 1.0 {
          rank 0 { local 0 device 0
            level 0 x TOPO_FILE_DESC { addr IPV4 10.0.0.1 ports [0/0] }
          }
        }
        """
        table = rt.parse_dsl(src)
        with self.assertRaises(rt.CheckError):
            rt.check(table)

    def test_rank_id_must_be_continuous(self) -> None:
        src = """
        ranktable 2.0 {
          rank 0 { local 0 device 0
            level 0 x TOPO_FILE_DESC { addr IPV4 10.0.0.1 ports [0/0] }
          }
          rank 2 { local 1 device 1
            level 0 x TOPO_FILE_DESC { addr IPV4 10.0.0.2 ports [0/1] }
          }
        }
        """
        with self.assertRaises(rt.CheckError) as ctx:
            rt.check(rt.parse_dsl(src))
        # size==2 so rank_id 2 is rejected as out of range first;
        # unique + [0, N) + count==N already implies continuity.
        self.assertIn("out of range", str(ctx.exception))

    def test_duplicate_rank_id(self) -> None:
        src = """
        ranktable 2.0 {
          rank 0 { local 0 device 0
            level 0 x TOPO_FILE_DESC { addr IPV4 10.0.0.1 ports [0/0] }
          }
          rank 0 { local 1 device 1
            level 0 x TOPO_FILE_DESC { addr IPV4 10.0.0.2 ports [0/1] }
          }
        }
        """
        with self.assertRaises(rt.CheckError):
            rt.check(rt.parse_dsl(src))

    def test_level_must_increase(self) -> None:
        src = """
        ranktable 2.0 {
          rank 0 { local 0 device 0
            level 1 x CLOS { addr IPV4 10.0.0.1 ports [0/0] }
            level 0 y TOPO_FILE_DESC { addr IPV4 10.0.0.2 ports [0/1] }
          }
        }
        """
        with self.assertRaises(rt.CheckError) as ctx:
            rt.check(rt.parse_dsl(src))
        self.assertIn("not increased", str(ctx.exception))

    def test_same_instance_addr_size(self) -> None:
        src = """
        ranktable 2.0 {
          rank 0 { local 0 device 0
            level 0 inst TOPO_FILE_DESC {
              addr IPV4 10.0.0.1 ports [0/0]
              addr IPV4 10.0.0.2 ports [0/1]
            }
          }
          rank 1 { local 1 device 1
            level 0 inst TOPO_FILE_DESC {
              addr IPV4 10.0.0.3 ports [0/0]
            }
          }
        }
        """
        with self.assertRaises(rt.CheckError) as ctx:
            rt.check(rt.parse_dsl(src))
        self.assertIn("rank_addrs size", str(ctx.exception))

    def test_v1_interleaved_ranks(self) -> None:
        src = """
        ranktable 1.0 {
          server a {
            device 0 ip 10.0.0.1 rank 0
            device 1 ip 10.0.0.2 rank 2
          }
          server b {
            device 0 ip 10.0.1.1 rank 1
            device 1 ip 10.0.1.2 rank 3
          }
        }
        """
        with self.assertRaises(rt.CheckError) as ctx:
            rt.check(rt.parse_dsl(src))
        self.assertIn("interleave", str(ctx.exception))

    def test_backup_local_id(self) -> None:
        src = """
        ranktable 2.0 {
          rank 0 { local 0 device 0
            level 0 x TOPO_FILE_DESC { addr IPV4 10.0.0.1 ports [0/0] }
          }
          rank 1 { local 64 replaced_local 2 device 1
            level 0 x TOPO_FILE_DESC { addr IPV4 10.0.0.2 ports [0/1] }
          }
        }
        """
        table = rt.parse_dsl(src)
        rt.check(table)
        self.assertEqual(table.ranks[1].replaced_local_id, 2)


class TestCli(unittest.TestCase):
    def test_compile_decompile_cli(self) -> None:
        src = EXAMPLES / "v2_single_layer.rt"
        with tempfile.TemporaryDirectory() as tmp:
            out_json = Path(tmp) / "out.json"
            out_rt = Path(tmp) / "out.rt"
            self.assertEqual(rt.main(["compile", str(src), "-o", str(out_json)]), 0)
            data = json.loads(out_json.read_text())
            self.assertEqual(data["version"], "2.0")
            self.assertEqual(data["rank_count"], 2)
            self.assertEqual(rt.main(["decompile", str(out_json), "-o", str(out_rt)]), 0)
            self.assertEqual(rt.main(["check", str(out_rt)]), 0)
            self.assertIn("rank 0", out_rt.read_text())


if __name__ == "__main__":
    unittest.main()
