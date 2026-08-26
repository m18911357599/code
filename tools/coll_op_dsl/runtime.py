"""Collective primitives: ring / recursive doubling / hierarchical + event log."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

ReduceFn = Callable[[int, int], int]

REDUCE_OPS: dict[str, ReduceFn] = {
    "SUM": lambda a, b: a + b,
    "MAX": max,
    "MIN": min,
}


@dataclass
class Event:
    step: int
    phase: str
    rank: int
    action: str
    peer: Optional[int]
    chunk: Optional[int]
    detail: str = ""

    def format(self) -> str:
        peer = f" rank{self.peer}" if self.peer is not None else ""
        chunk = f" chunk{self.chunk}" if self.chunk is not None else ""
        extra = f" {self.detail}" if self.detail else ""
        return (
            f"  [{self.phase} s{self.step}] rank{self.rank} {self.action}{peer}{chunk}{extra}"
        )


@dataclass
class Trace:
    events: list[Event] = field(default_factory=list)
    step: int = 0

    def emit(
        self,
        phase: str,
        rank: int,
        action: str,
        peer: Optional[int] = None,
        chunk: Optional[int] = None,
        detail: str = "",
    ) -> None:
        self.events.append(Event(self.step, phase, rank, action, peer, chunk, detail))

    def bump(self) -> None:
        self.step += 1

    def dump(self, rank_filter: Optional[int] = None) -> str:
        lines = []
        for ev in self.events:
            if rank_filter is not None and ev.rank != rank_filter:
                continue
            lines.append(ev.format())
        return "\n".join(lines)


def _require_divisible(count: int, n: int, what: str) -> int:
    if n <= 0:
        raise ValueError(f"{what}: empty group")
    if count % n != 0:
        raise ValueError(f"{what}: count {count} not divisible by group size {n}")
    return count // n


def owned_after_ring_rs(index: int, n: int) -> int:
    return (index + 1) % n


def ring_reduce_scatter(
    data: dict[int, list[int]], group: list[int], op: ReduceFn, trace: Trace, phase: str = "reducescatter"
) -> dict[int, list[int]]:
    n = len(group)
    count = len(data[group[0]])
    chunk = _require_divisible(count, n, "ReduceScatter")
    buf = {r: list(data[r]) for r in group}
    for t in range(n - 1):
        incoming: dict[int, list[int]] = {}
        recv_at: dict[int, int] = {}
        for i, r in enumerate(group):
            nxt = group[(i + 1) % n]
            prev = group[(i - 1) % n]
            send_idx = (i - t) % n
            recv_idx = (i - t - 1) % n
            send_sl = slice(send_idx * chunk, (send_idx + 1) * chunk)
            recv_sl = slice(recv_idx * chunk, (recv_idx + 1) * chunk)
            incoming[r] = buf[prev][recv_sl]
            recv_at[r] = recv_idx
            trace.emit(phase, r, "send", nxt, send_idx)
            trace.emit(phase, r, "recv", prev, recv_idx)
        for r in group:
            idx = recv_at[r]
            sl = slice(idx * chunk, (idx + 1) * chunk)
            buf[r][sl] = [op(a, b) for a, b in zip(buf[r][sl], incoming[r])]
            trace.emit(phase, r, "reduce", None, idx)
        trace.bump()
    return buf


def ring_allgather(
    data: dict[int, list[int]],
    group: list[int],
    trace: Trace,
    owned: Optional[Callable[[int, int], int]] = None,
    phase: str = "allgather",
) -> dict[int, list[int]]:
    n = len(group)
    count = len(data[group[0]])
    chunk = _require_divisible(count, n, "AllGather")
    if owned is None:
        owned = lambda i, n: i  # noqa: E731
    buf = {r: list(data[r]) for r in group}
    for t in range(n - 1):
        incoming: dict[int, tuple[int, list[int]]] = {}
        for i, r in enumerate(group):
            nxt = group[(i + 1) % n]
            prev = group[(i - 1) % n]
            send_idx = (owned(i, n) - t) % n
            recv_idx = (owned(i, n) - t - 1) % n
            send_sl = slice(send_idx * chunk, (send_idx + 1) * chunk)
            recv_sl = slice(recv_idx * chunk, (recv_idx + 1) * chunk)
            incoming[r] = (recv_idx, buf[prev][recv_sl])
            trace.emit(phase, r, "send", nxt, send_idx)
            trace.emit(phase, r, "recv", prev, recv_idx)
        for r, (idx, piece) in incoming.items():
            sl = slice(idx * chunk, (idx + 1) * chunk)
            buf[r][sl] = list(piece)
            trace.emit(phase, r, "copy", None, idx)
        trace.bump()
    return buf


def ring_allreduce(
    data: dict[int, list[int]],
    group: list[int],
    op: ReduceFn,
    trace: Trace,
    rs_phase: str = "reducescatter",
    ag_phase: str = "allgather",
) -> dict[int, list[int]]:
    buf = ring_reduce_scatter(data, group, op, trace, phase=rs_phase)
    return ring_allgather(buf, group, trace, owned=owned_after_ring_rs, phase=ag_phase)


def ring_broadcast(
    data: dict[int, list[int]], group: list[int], root: int, trace: Trace
) -> dict[int, list[int]]:
    if root not in group:
        raise ValueError(f"root {root} not in group {group}")
    n = len(group)
    buf = {r: list(data[r]) for r in group}
    payload = list(buf[root])
    root_i = group.index(root)
    for hop in range(1, n):
        src_i = (root_i + hop - 1) % n
        dst_i = (root_i + hop) % n
        src, dst = group[src_i], group[dst_i]
        buf[dst] = list(payload if src == root else buf[src])
        trace.emit("broadcast", src, "send", dst, None)
        trace.emit("broadcast", dst, "recv", src, None, "copy")
        trace.bump()
    for r in group:
        buf[r] = list(buf[root] if r == root else buf[r])
    # ensure everyone equals root
    for r in group:
        buf[r] = list(buf[root])
    return buf


def ring_reduce(
    data: dict[int, list[int]], group: list[int], root: int, op: ReduceFn, trace: Trace
) -> dict[int, list[int]]:
    """Ring reduce into root; other ranks keep their (unspecified) buffers."""
    if root not in group:
        raise ValueError(f"root {root} not in group {group}")
    n = len(group)
    buf = {r: list(data[r]) for r in group}
    root_i = group.index(root)
    acc = list(buf[root])
    for hop in range(1, n):
        src_i = (root_i - hop) % n
        src = group[src_i]
        acc = [op(a, b) for a, b in zip(acc, buf[src])]
        trace.emit("reduce", src, "send", root, None)
        trace.emit("reduce", root, "recv", src, None, "reduce")
        trace.bump()
    out = {r: list(buf[r]) for r in group}
    out[root] = acc
    return out


def rec_dbl_allreduce(
    data: dict[int, list[int]], group: list[int], op: ReduceFn, trace: Trace
) -> dict[int, list[int]]:
    n = len(group)
    if n & (n - 1):
        raise ValueError(f"recursive_hd requires power-of-two group, got {n}")
    buf = {r: list(data[r]) for r in group}
    rounds = n.bit_length() - 1
    for k in range(rounds):
        nxt = {}
        for i, r in enumerate(group):
            partner = group[i ^ (1 << k)]
            nxt[r] = [op(a, b) for a, b in zip(buf[r], buf[partner])]
            trace.emit("rec_dbl", r, "xchg", partner, None, f"round{k}")
            trace.emit("rec_dbl", r, "reduce", partner, None, f"round{k}")
        buf = nxt
        trace.bump()
    return buf


def hierarchical_allreduce(
    data: dict[int, list[int]],
    intra_groups: list[list[int]],
    slot_groups: list[list[int]],
    op: ReduceFn,
    trace: Trace,
) -> dict[int, list[int]]:
    if not intra_groups:
        raise ValueError("hierarchical needs intra groups")
    n0 = len(intra_groups[0])
    if any(len(g) != n0 for g in intra_groups):
        raise ValueError("hierarchical intra groups must be equal width")
    buf = {r: list(v) for r, v in data.items()}
    for g in intra_groups:
        buf.update(ring_reduce_scatter(buf, g, op, trace, phase="intra_rs"))
    count = len(buf[intra_groups[0][0]])
    chunk = count // n0
    for slot, group in enumerate(slot_groups):
        owned = owned_after_ring_rs(slot, n0)
        sl = slice(owned * chunk, (owned + 1) * chunk)
        sub = {r: buf[r][sl] for r in group}
        # AllReduce on slice: treat slice as the whole buffer of length `chunk`.
        # If inter group size does not divide chunk, use rec_dbl if power of 2 else
        # pad-free element-wise ring by replicating as 1-chunk allreduce via rec/ring on length.
        sub_full = _small_allreduce(sub, group, op, trace, phase="inter_ar")
        for r in group:
            buf[r][sl] = sub_full[r]
    for g in intra_groups:
        buf.update(ring_allgather(buf, g, trace, owned=owned_after_ring_rs, phase="intra_ag"))
    return buf


def _small_allreduce(
    data: dict[int, list[int]], group: list[int], op: ReduceFn, trace: Trace, phase: str
) -> dict[int, list[int]]:
    """AllReduce a buffer whose length may not be divisible by |group|."""
    n = len(group)
    count = len(data[group[0]])
    if n == 1:
        return {r: list(data[r]) for r in group}
    if count % n == 0:
        tmp = ring_allreduce(data, group, op, trace, rs_phase=f"{phase}_rs", ag_phase=f"{phase}_ag")
        return tmp
    if (n & (n - 1)) == 0:
        return rec_dbl_allreduce(data, group, op, trace)
    # fallback: serialize reduce to group[0] then broadcast
    acc = list(data[group[0]])
    for r in group[1:]:
        acc = [op(a, b) for a, b in zip(acc, data[r])]
        trace.emit(phase, r, "send", group[0], None)
        trace.emit(phase, group[0], "recv", r, None, "reduce")
        trace.bump()
    out = {r: list(acc) for r in group}
    for r in group[1:]:
        trace.emit(phase, group[0], "send", r, None)
        trace.emit(phase, r, "recv", group[0], None, "copy")
        trace.bump()
    return out


def rs_output_owned(buf: dict[int, list[int]], group: list[int]) -> dict[int, list[int]]:
    n = len(group)
    count = len(buf[group[0]])
    chunk = count // n
    out = {}
    for i, r in enumerate(group):
        owned = owned_after_ring_rs(i, n)
        sl = slice(owned * chunk, (owned + 1) * chunk)
        out[r] = list(buf[r][sl])
    return out

