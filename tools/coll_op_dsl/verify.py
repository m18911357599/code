#!/usr/bin/env python3
"""End-to-end verification of RankTable → topo model → algo template → operator."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import coll_op_dsl as cli
from topo_model import load_topo

RT = ROOT.parent / "ranktable_dsl" / "examples" / "v2_two_layer.rt"
EX = ROOT / "examples"


def section(title: str) -> None:
    print()
    print("=" * 64)
    print(title)
    print("=" * 64)


def main() -> int:
    section("1) RankTable DSL -> Topology Model")
    topo = load_topo(RT)
    print(topo.describe())
    print("queries:")
    print("  layers()", topo.layer_ids())
    print("  instances(0)", [(i.inst_id, i.ranks) for i in topo.instances(0)])
    print("  slot_groups(0)", topo.slot_groups(0))

    section("2) Same operator AllReduce, two algorithm templates")
    ring_p, _, _ = cli._load_prog_and_topo(str(EX / "allreduce_ring.coll"))
    hier_p, _, _ = cli._load_prog_and_topo(str(EX / "allreduce_hier.coll"))
    ring = cli.verify_operator(ring_p, topo, ring_p.operators[0])
    hier = cli.verify_operator(hier_p, topo, hier_p.operators[0])
    print(f"ring         PASS={ring['ok']} steps={ring['steps']} events={ring['events']}")
    print(f"hierarchical PASS={hier['ok']} steps={hier['steps']} events={hier['events']}")
    print("outputs equal:", ring["outputs"] == hier["outputs"])
    print("schedule equal:", ring["trace"].dump() == hier["trace"].dump())
    print("gold AllReduce SUM (rank0):", ring["outputs"][0])
    print()
    print("--- ring schedule (rank 0) ---")
    print(ring["trace"].dump(0))
    print()
    print("--- hierarchical schedule (rank 0) ---")
    print(hier["trace"].dump(0))
    if not ring["ok"] or not hier["ok"] or ring["outputs"] != hier["outputs"]:
        print("FAIL: operator result mismatch")
        return 1
    if ring["trace"].dump() == hier["trace"].dump():
        print("FAIL: expected different schedules")
        return 1

    section("3) Other operators on the same topology")
    rc = 0
    for name in ("reducescatter_ring.coll", "allgather_ring.coll", "broadcast_ring.coll", "allreduce_rec.coll"):
        code = cli.main(["sim", str(EX / name)])
        rc |= code

    section("4) V1 server table uses the same operator+template")
    rc |= cli.main(["sim", str(EX / "allreduce_v1.coll")])

    section("5) match rejects recursive_hd on 3 ranks")
    src = """
    topo from "three_ranks.rt"
    operator AllReduce { reduce SUM count 4 use recursive_hd on layer 0 }
    """
    from program import parse_coll

    prog = parse_coll(src)
    three = load_topo(ROOT / "tests" / "fixtures" / "three_ranks.rt")
    try:
        cli.run_operator(prog, three, prog.operators[0])
        print("FAIL: expected MatchError")
        return 1
    except cli.MatchError as exc:
        print("rejected:", exc)

    print()
    print("verify: all checks passed" if rc == 0 else "verify: some sims failed")
    return rc


if __name__ == "__main__":
    sys.exit(main())
