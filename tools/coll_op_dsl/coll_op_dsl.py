#!/usr/bin/env python3
"""CLI: ranktable → topology model → algorithm template → operator instance."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from program import AlgoDef, CollSyntaxError, OperatorDef, Program, parse_coll, parse_coll_file
from runtime import (
    REDUCE_OPS,
    Trace,
    hierarchical_allreduce,
    rec_dbl_allreduce,
    ring_allgather,
    ring_allreduce,
    ring_broadcast,
    ring_reduce,
    ring_reduce_scatter,
    rs_output_owned,
)
from topo_model import TopoModel, load_topo

OPS = {"AllReduce", "ReduceScatter", "AllGather", "Broadcast", "Reduce"}

_SEARCH = [
    Path(__file__).resolve().parent / "examples",
    Path(__file__).resolve().parent / "tests" / "fixtures",
    Path(__file__).resolve().parent.parent / "ranktable_dsl" / "examples",
    Path(__file__).resolve().parent.parent / "ranktable_dsl" / "tests" / "fixtures",
]


def resolve_topo(spec: str, coll_path: Optional[Path] = None) -> Path:
    p = Path(spec)
    candidates = []
    if p.is_file():
        return p
    if coll_path:
        candidates.append(coll_path.parent / spec)
    candidates.append(Path.cwd() / spec)
    candidates.extend(d / spec for d in _SEARCH)
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(f"rank table not found: {spec}")


class MatchError(Exception):
    pass


def check_match(algo: AlgoDef, topo: TopoModel, group_size: int) -> None:
    m = algo.match
    if m.ranks_ge is not None and group_size < m.ranks_ge:
        raise MatchError(f"algo {algo.name}: ranks {group_size} < {m.ranks_ge}")
    if m.layers_ge is not None and len(topo.layer_ids()) < m.layers_ge:
        raise MatchError(f"algo {algo.name}: layers {len(topo.layer_ids())} < {m.layers_ge}")
    if m.power_of_two and group_size & (group_size - 1):
        raise MatchError(f"algo {algo.name}: group size {group_size} is not power of two")
    if algo.pattern == "hierarchical":
        if len(topo.layer_ids()) < 2:
            raise MatchError(f"algo {algo.name}: need >= 2 layers")
        insts = topo.instances(algo.intra_layer)
        if not insts:
            raise MatchError(f"algo {algo.name}: no instances on layer {algo.intra_layer}")
        w = len(insts[0].ranks)
        if any(len(i.ranks) != w for i in insts):
            raise MatchError(f"algo {algo.name}: intra instances not equal width")


def groups_for_flat(topo: TopoModel, layer: Optional[int]) -> list[list[int]]:
    if not topo.layer_ids():
        return [list(topo.rank_ids)]
    L = layer if layer is not None else topo.layer_ids()[-1]
    return [list(inst.ranks) for inst in topo.instances(L)]


def seed_buffers(ranks: list[int], count: int, kind: str) -> dict[int, list[int]]:
    if kind == "AllGather":
        return {r: [r] * count for r in ranks}
    if kind == "Broadcast":
        return {r: ([100 + i for i in range(count)] if r == ranks[0] else [0] * count) for r in ranks}
    return {r: [r * count + i for i in range(count)] for r in ranks}


def expected_allreduce(bufs: dict[int, list[int]], ranks: list[int], opname: str) -> list[int]:
    op = REDUCE_OPS[opname]
    acc = list(bufs[ranks[0]])
    for r in ranks[1:]:
        acc = [op(a, b) for a, b in zip(acc, bufs[r])]
    return acc


def run_flat(
    op: OperatorDef,
    pattern: str,
    groups: list[list[int]],
    bufs: dict[int, list[int]],
    trace: Trace,
) -> dict[int, list[int]]:
    fn = REDUCE_OPS[op.reduce]
    out = {r: list(v) for r, v in bufs.items()}
    for g in groups:
        sub = {r: out[r] for r in g}
        if op.name == "AllReduce":
            if pattern == "recursive_hd":
                got = rec_dbl_allreduce(sub, g, fn, trace)
            else:
                got = ring_allreduce(sub, g, fn, trace)
            out.update(got)
        elif op.name == "ReduceScatter":
            rs = ring_reduce_scatter(sub, g, fn, trace)
            out.update(rs_output_owned(rs, g))
        elif op.name == "AllGather":
            n = len(g)
            chunk = op.count
            expanded = {}
            for i, r in enumerate(g):
                row = [0] * (chunk * n)
                row[i * chunk : (i + 1) * chunk] = list(sub[r])
                expanded[r] = row
            got = ring_allgather(expanded, g, trace)
            out.update(got)
        elif op.name == "Broadcast":
            out.update(ring_broadcast(sub, g, op.root if op.root in g else g[0], trace))
        elif op.name == "Reduce":
            root = op.root if op.root in g else g[0]
            out.update(ring_reduce(sub, g, root, fn, trace))
        else:
            raise ValueError(f"unsupported operator {op.name}")
    return out


def run_operator(prog: Program, topo: TopoModel, op: OperatorDef) -> tuple[dict[int, list[int]], Trace, dict]:
    if op.name not in OPS:
        raise ValueError(f"unknown operator {op.name}")
    if op.reduce not in REDUCE_OPS:
        raise ValueError(f"unknown reduce {op.reduce}")
    algo = prog.algos.get(op.use)
    if algo is None:
        raise ValueError(f"unknown algo {op.use}")

    ranks = topo.rank_ids
    bufs = seed_buffers(ranks, op.count, op.name)
    trace = Trace()
    meta = {"op": op.name, "algo": op.use, "pattern": algo.pattern, "groups": []}

    if algo.pattern == "hierarchical":
        if op.name != "AllReduce":
            raise MatchError(f"hierarchical currently lowers AllReduce only, not {op.name}")
        check_match(algo, topo, len(ranks))
        intra = [list(i.ranks) for i in topo.instances(algo.intra_layer)]
        slots = topo.slot_groups(algo.intra_layer)
        meta["groups"] = {"intra": intra, "slots": slots}
        out = hierarchical_allreduce(bufs, intra, slots, REDUCE_OPS[op.reduce], trace)
        return out, trace, meta

    groups = groups_for_flat(topo, op.layer)
    for g in groups:
        check_match(algo, topo, len(g))
    meta["groups"] = groups
    out = run_flat(op, algo.pattern, groups, bufs, trace)
    return out, trace, meta


def verify_operator(prog: Program, topo: TopoModel, op: OperatorDef) -> dict:
    ranks = topo.rank_ids
    inputs = seed_buffers(ranks, op.count, op.name)
    out, trace, meta = run_operator(prog, topo, op)
    fn = REDUCE_OPS[op.reduce]
    ok = True
    detail = ""

    if op.name == "AllReduce":
        if meta["pattern"] == "hierarchical" or (isinstance(meta["groups"], list) and len(meta["groups"]) == 1):
            gold = expected_allreduce(inputs, ranks, op.reduce)
            for r in ranks:
                if out[r] != gold:
                    ok = False
                    detail = f"rank {r} {out[r]} != {gold}"
                    break
        else:
            for g in meta["groups"]:
                gold = expected_allreduce(inputs, g, op.reduce)
                for r in g:
                    if out[r] != gold:
                        ok = False
                        detail = f"rank {r} {out[r]} != {gold}"
                        break
    elif op.name == "ReduceScatter":
        for g in meta["groups"]:
            n = len(g)
            gold_full = expected_allreduce(inputs, g, op.reduce)
            chunk = op.count // n
            for i, r in enumerate(g):
                owned = (i + 1) % n
                exp = gold_full[owned * chunk : (owned + 1) * chunk]
                if out[r] != exp:
                    ok = False
                    detail = f"rank {r} {out[r]} != {exp}"
                    break
    elif op.name == "AllGather":
        for g in meta["groups"]:
            gold = []
            for r in g:
                gold.extend(inputs[r])
            for r in g:
                if out[r] != gold:
                    ok = False
                    detail = f"rank {r} {out[r]} != {gold}"
                    break
    elif op.name == "Broadcast":
        for g in meta["groups"]:
            root = op.root if op.root in g else g[0]
            gold = inputs[root]
            for r in g:
                if out[r] != gold:
                    ok = False
                    detail = f"rank {r} {out[r]} != {gold}"
                    break
    elif op.name == "Reduce":
        for g in meta["groups"]:
            root = op.root if op.root in g else g[0]
            gold = expected_allreduce(inputs, g, op.reduce)
            if out[root] != gold:
                ok = False
                detail = f"root {root} {out[root]} != {gold}"

    return {
        "ok": ok,
        "detail": detail,
        "op": op.name,
        "algo": op.use,
        "pattern": meta["pattern"],
        "inputs": inputs,
        "outputs": out,
        "trace": trace,
        "meta": meta,
        "steps": trace.step,
        "events": len(trace.events),
    }


def _load_prog_and_topo(coll: str) -> tuple[Program, TopoModel, Path]:
    path = Path(coll)
    prog = parse_coll_file(path)
    if not prog.topo_from:
        raise ValueError("program needs 'topo from \"...\"'")
    topo_path = resolve_topo(prog.topo_from, path)
    return prog, load_topo(topo_path), topo_path


def cmd_topo(args: argparse.Namespace) -> int:
    model = load_topo(args.input)
    text = model.describe()
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


def cmd_lower(args: argparse.Namespace) -> int:
    prog, topo, _ = _load_prog_and_topo(args.input)
    for op in prog.operators:
        _, trace, meta = run_operator(prog, topo, op)
        print(f"# {op.name} use {op.use} pattern={meta['pattern']} steps={trace.step}")
        print(trace.dump(args.rank))
        print()
    return 0


def cmd_sim(args: argparse.Namespace) -> int:
    prog, topo, _ = _load_prog_and_topo(args.input)
    rc = 0
    for op in prog.operators:
        result = verify_operator(prog, topo, op)
        status = "PASS" if result["ok"] else "FAIL"
        print(
            f"{status} {op.name} algo={op.use} steps={result['steps']} events={result['events']}"
        )
        if args.verbose:
            print("  inputs ", result["inputs"])
            print("  outputs", result["outputs"])
        if not result["ok"]:
            print(" ", result["detail"])
            rc = 1
    return rc


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Topology model + collective operator DSL")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("topo", help="RankTable -> topology model DSL")
    t.add_argument("input")
    t.add_argument("-o", "--output", default=None)

    lo = sub.add_parser("lower", help="print lowered schedule")
    lo.add_argument("input")
    lo.add_argument("--rank", type=int, default=None)

    s = sub.add_parser("sim", help="simulate and check collective semantics")
    s.add_argument("input")
    s.add_argument("-v", "--verbose", action="store_true")

    args = p.parse_args(argv)
    try:
        return {"topo": cmd_topo, "lower": cmd_lower, "sim": cmd_sim}[args.cmd](args)
    except (CollSyntaxError, MatchError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
