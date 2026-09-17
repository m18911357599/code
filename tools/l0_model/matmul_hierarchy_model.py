#!/usr/bin/env python3
"""Software model: why Ascend Cube needs L0A/L0B on top of L1.

Tile-level (not RTL-accurate). Captures:

1. Multi-core split of A(M) or B(N) that turns GEMM 搬入-bound.
2. GM->L1 bursts of m*ka and kb*n cached in L1.
3. Reload amplification if Cube reads L1 at native 16x16 (no L0).
4. Asymmetric L1->L0A=512 B/cyc vs L1->L0B=256 B/cyc (910B1).
5. 2-/3-stage overlap, plus a pessimistic L1 shared-port model.

Default numbers: public Ascend 910B1 platform_config.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


CUBE = 16
ALIGNMENTS = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096)


@dataclass(frozen=True)
class Arch:
    name: str = "Ascend910B1"
    cube_m: int = 16
    cube_n: int = 16
    cube_k: int = 16
    macs_per_cycle: int = 4096
    l1_size: int = 512 * 1024
    l0a_size: int = 64 * 1024
    l0b_size: int = 64 * 1024
    l0c_size: int = 128 * 1024
    bw_l1_l0a: float = 512.0
    bw_l1_l0b: float = 256.0
    bw_gm_l1: float = 32.0
    l1_latency: int = 16
    gm_latency: int = 120
    dtype_in: int = 2
    dtype_acc: int = 4
    pingpong_l1: bool = True
    pingpong_l0: bool = True

    @property
    def cube_a_bw(self) -> float:
        return self.macs_per_cycle * self.dtype_in / self.cube_n

    @property
    def cube_b_bw(self) -> float:
        return self.macs_per_cycle * self.dtype_in / self.cube_m

    @property
    def compute_bound_pair_size(self) -> float:
        """mn/(m+n) needed to be compute-bound vs HBM."""
        return self.macs_per_cycle * self.dtype_in / self.bw_gm_l1

    @property
    def min_n_for_l0a_fill(self) -> float:
        """n_l0 so L1->L0A=512 B/cyc hides behind Cube (A-fill)."""
        return self.macs_per_cycle * self.dtype_in / self.bw_l1_l0a

    @property
    def min_m_for_l0b_fill(self) -> float:
        """m_l0 so L1->L0B=256 B/cyc hides behind Cube (B-fill)."""
        return self.macs_per_cycle * self.dtype_in / self.bw_l1_l0b

    @property
    def min_n_for_l0b_fill(self) -> float:
        # kept for older call sites; B-fill hide is on m, see min_m_for_l0b_fill.
        return self.min_m_for_l0b_fill


@dataclass(frozen=True)
class Hierarchy:
    has_l0a: bool = True
    has_l0b: bool = True
    l0a_size: Optional[int] = None
    l0b_size: Optional[int] = None
    l1_size: Optional[int] = None
    shared_port: bool = False

    def label(self) -> str:
        parts = []
        if self.has_l0a:
            parts.append("L0A")
        if self.has_l0b:
            parts.append("L0B")
        name = "+".join(parts) if parts else "L1-only"
        if self.shared_port:
            name += "/sharedL1"
        return name


@dataclass(frozen=True)
class Tile:
    m_blk: int
    n_blk: int
    k_l1: int
    m_l0: int
    n_l0: int
    k_l0: int
    stationary: str


@dataclass
class CoreResult:
    m: int
    n: int
    k: int
    tile: Optional[Tile]
    cycles: float
    t_cube: float
    t_gm: float
    t_mte1: float
    t_mte1_a: float
    t_mte1_b: float
    bound: str
    cube_util: float
    gm_util: float
    a_reload: float
    b_reload: float
    l0a_over_l1: float
    l0b_over_l1: float
    l1_working_bytes: int
    l0a_working_bytes: int
    l0b_working_bytes: int
    intensity: float
    hierarchy: str
    notes: str = ""


def _align_down(x: int, a: int = CUBE) -> int:
    return max(a, (int(x) // a) * a)


def _align_up(x: int, a: int = CUBE) -> int:
    return max(a, int(math.ceil(x / a) * a))


def _cands(limit: int) -> List[int]:
    vals = [a for a in ALIGNMENTS if a <= limit]
    if limit >= CUBE:
        vals.append(_align_down(limit))
    return sorted(set(v for v in vals if v >= CUBE))


def _budget(size: int, pingpong: bool) -> int:
    return size // 2 if pingpong else size


def split_shape(m: int, n: int, k: int, block_num: int, mode: str) -> Tuple[int, int, int]:
    block_num = max(1, int(block_num))
    if mode == "A":
        return min(m, _align_up(math.ceil(m / block_num))), n, k
    if mode == "B":
        return m, min(n, _align_up(math.ceil(n / block_num))), k
    if mode == "K":
        return m, n, min(k, _align_up(math.ceil(k / block_num)))
    if mode == "2D":
        g_m = int(math.floor(math.sqrt(block_num)))
        while g_m > 1 and block_num % g_m != 0:
            g_m -= 1
        g_n = block_num // max(g_m, 1)
        return (
            min(m, _align_up(math.ceil(m / max(g_m, 1)))),
            min(n, _align_up(math.ceil(n / max(g_n, 1)))),
            k,
        )
    raise ValueError(mode)


def is_load_bound(m: int, n: int, arch: Arch) -> bool:
    return (m * n) / max(m + n, 1) < arch.compute_bound_pair_size


def pick_l0_shape(arch: Arch, hier: Hierarchy, m: int, n: int, k: int) -> Tuple[int, int, int, str]:
    s, acc = arch.dtype_in, arch.dtype_acc
    ba = _budget(hier.l0a_size or arch.l0a_size, arch.pingpong_l0) if hier.has_l0a else s * CUBE * CUBE
    bb = _budget(hier.l0b_size or arch.l0b_size, arch.pingpong_l0) if hier.has_l0b else s * CUBE * CUBE
    bc = _budget(arch.l0c_size, arch.pingpong_l0)
    l1_budget = _budget(hier.l1_size or arch.l1_size, arch.pingpong_l1)
    m_lim = min(m, 2048)
    n_lim = min(n, 2048)
    k_lim = min(k, 2048)
    best = (arch.cube_m, arch.cube_n, arch.cube_k, "A")
    best_score = None
    m_cands = _cands(m_lim if hier.has_l0a else arch.cube_m)
    n_cands = _cands(n_lim if hier.has_l0b else arch.cube_n)
    for m0 in m_cands:
        for n0 in n_cands:
            for k0 in _cands(k_lim):
                if m0 * n0 * acc > bc:
                    continue
                if hier.has_l0a and m0 * k0 * s > ba:
                    continue
                if hier.has_l0b and k0 * n0 * s > bb:
                    continue
                if not hier.has_l0a and m0 != arch.cube_m:
                    continue
                if not hier.has_l0b and n0 != arch.cube_n:
                    continue
                if (min(m, m0) + min(n, n0)) * k0 * s > l1_budget:
                    continue
                stats = []
                if hier.has_l0a and not hier.has_l0b:
                    stats = ["A"]
                elif hier.has_l0b and not hier.has_l0a:
                    stats = ["B"]
                else:
                    stats = ["A", "B"]
                for stationary in stats:
                    a_reload = 1.0 if (hier.has_l0a and stationary == "A") else math.ceil(n / n0)
                    b_reload = 1.0 if (hier.has_l0b and stationary == "B") else math.ceil(m / m0)
                    t_a = (m * k * s * a_reload) / arch.bw_l1_l0a
                    t_b = (n * k * s * b_reload) / arch.bw_l1_l0b
                    t_mte1 = t_a + t_b if not (hier.has_l0a or hier.has_l0b) else max(t_a, t_b)
                    t_cube = (m * n * k) / arch.macs_per_cycle
                    # Prefer L0 tiles that still allow a long GM burst in L1.
                    denom = (m + n) * s
                    k1 = 0 if denom <= 0 else (l1_budget // denom // k0) * k0
                    if k1 < k0:
                        k1 = k0
                        n_l1 = math.ceil(m / m0) * math.ceil(n / n0) * math.ceil(k / k0)
                    else:
                        n_l1 = math.ceil(k / k1)
                    t_gm = (m * k + n * k) * s / arch.bw_gm_l1 + n_l1 * arch.gm_latency
                    hide_b = min(m0 / arch.min_m_for_l0b_fill, 1.0) if hier.has_l0b else 1.0
                    score = (-max(t_cube, t_gm, t_mte1), -t_mte1, hide_b, k1, m0 * n0 * k0)
                    if best_score is None or score > best_score:
                        best_score = score
                        best = (m0, n0, k0, stationary)
    return best


def pick_l1_panel(arch: Arch, hier: Hierarchy, m: int, n: int, k: int, m0: int, n0: int, k0: int) -> Tuple[int, int, int]:
    """Choose L1 (m_blk, n_blk, k_l1) holding m*ka and kb*n ping-pong buffers."""
    s = arch.dtype_in
    budget = _budget(hier.l1_size or arch.l1_size, arch.pingpong_l1)
    best = None
    best_score = None
    for m_blk in _cands(m):
        if m_blk % m0 != 0 and m_blk != m:
            continue
        for n_blk in _cands(n):
            if n_blk % n0 != 0 and n_blk != n:
                continue
            denom = (m_blk + n_blk) * s
            if denom <= 0:
                continue
            raw = budget // denom
            step = k0 if k0 >= CUBE else CUBE
            k_l1 = min(k, (raw // step) * step)
            if k_l1 < step:
                continue
            n_panels = math.ceil(m / m_blk) * math.ceil(n / n_blk) * math.ceil(k / k_l1)
            intensity = (m_blk * n_blk) / (m_blk + n_blk)
            score = (-n_panels, intensity, k_l1, m_blk + n_blk)
            if best_score is None or score > best_score:
                best_score = score
                best = (m_blk, n_blk, k_l1)
    if best is None:
        m_blk = min(m, m0)
        n_blk = min(n, n0)
        denom = max((m_blk + n_blk) * arch.dtype_in, 1)
        k_l1 = min(k, (budget // denom) // CUBE * CUBE)
        return m_blk, n_blk, max(0, k_l1)
    return best


def select_tile(arch: Arch, hier: Hierarchy, m: int, n: int, k: int) -> Optional[Tile]:
    m0, n0, k0, stationary = pick_l0_shape(arch, hier, m, n, k)
    m_blk, n_blk, k_l1 = pick_l1_panel(arch, hier, m, n, k, m0, n0, k0)
    if m_blk < CUBE or n_blk < CUBE or k_l1 < CUBE:
        return None
    return Tile(m_blk, n_blk, k_l1, m0, n0, k0, stationary)


def reload_factors(arch: Arch, hier: Hierarchy, tile: Tile, m: int, n: int) -> Tuple[float, float]:
    """Times each A/B element is read from L1 over the full per-core GEMM.

    Without L0A, Cube's A latch is 16xK_pulse, so A is re-fetched for every
    n-step of cube_n. Same for L0B vs m-step of cube_m.
    With L0, A (B) is reused across n_l0 (m_l0) and only re-read if the
    stationary operand cannot cover the full core n (m).
    """
    n_panel = tile.n_l0 if hier.has_l0b else arch.cube_n
    m_panel = tile.m_l0 if hier.has_l0a else arch.cube_m
    if hier.has_l0a and tile.stationary == "A":
        a_reload = 1.0
    else:
        a_reload = float(math.ceil(n / n_panel))
    if hier.has_l0b and tile.stationary == "B":
        b_reload = 1.0
    else:
        b_reload = float(math.ceil(m / m_panel))
    return a_reload, b_reload


def _n_fills(m: int, n: int, k: int, tile: Tile) -> Tuple[int, int, int]:
    n_m = math.ceil(m / tile.m_l0)
    n_n = math.ceil(n / tile.n_l0)
    n_k = math.ceil(k / tile.k_l0)
    if tile.stationary == "A":
        n_a = n_m * n_k
        n_b = n_m * n_n * n_k
    else:
        n_b = n_n * n_k
        n_a = n_m * n_n * n_k
    n_l1 = math.ceil(m / tile.m_blk) * math.ceil(n / tile.n_blk) * math.ceil(k / tile.k_l1)
    return n_a, n_b, n_l1


def resource_times(arch: Arch, hier: Hierarchy, m: int, n: int, k: int, tile: Tile) -> Tuple[float, float, float, float, float]:
    s = arch.dtype_in
    a_reload, b_reload = reload_factors(arch, hier, tile, m, n)
    t_cube = (m * n * k) / arch.macs_per_cycle
    gm_bytes = (m * k + n * k) * s
    n_a, n_b, n_l1 = _n_fills(m, n, k, tile)
    t_gm = gm_bytes / arch.bw_gm_l1 + n_l1 * arch.gm_latency

    bytes_a = m * k * s * a_reload
    bytes_b = n * k * s * b_reload
    t_a = bytes_a / arch.bw_l1_l0a + n_a * arch.l1_latency
    t_b = bytes_b / arch.bw_l1_l0b + n_b * arch.l1_latency

    no_l0 = not (hier.has_l0a or hier.has_l0b)
    if hier.shared_port or no_l0:
        t_mte1 = t_a + t_b
    else:
        t_mte1 = max(t_a, t_b)

    # Without L0, Cube A+B ports must be served from L1. Even with reload=1
    # the native 16x16 issue needs 512+512 B/cyc; L1 only has 512+256.
    if not hier.has_l0a or not hier.has_l0b:
        req_a = t_cube if not hier.has_l0a else 0.0
        req_b = t_cube if not hier.has_l0b else 0.0
        # Cap: L1 cannot issue Cube pulses faster than its ports allow.
        # Each pulse needs cube_m*cube_k*s from A and cube_k*cube_n*s from B.
        pulse = (m / arch.cube_m) * (n / arch.cube_n) * (k / arch.cube_k)
        t_port_a = pulse * (arch.cube_m * arch.cube_k * s) / arch.bw_l1_l0a if not hier.has_l0a else 0.0
        t_port_b = pulse * (arch.cube_k * arch.cube_n * s) / arch.bw_l1_l0b if not hier.has_l0b else 0.0
        if hier.shared_port or no_l0:
            t_mte1 = max(t_mte1, t_port_a + t_port_b)
        else:
            t_mte1 = max(t_mte1, t_a, t_b, t_port_a, t_port_b)
        _ = (req_a, req_b)

    prologue = arch.gm_latency + (tile.m_blk * tile.k_l1 + tile.k_l1 * tile.n_blk) * s / arch.bw_gm_l1
    if hier.has_l0a or hier.has_l0b:
        prologue += arch.l1_latency + tile.m_l0 * tile.k_l0 * s / arch.bw_l1_l0a
    else:
        prologue += arch.l1_latency

    # Shared-port 搬入 path: Cube L1 reads steal the only R/W resource from GM.
    if hier.shared_port and no_l0:
        t_gm = t_gm + 0.5 * min(t_gm, t_mte1)

    t_total = max(t_cube, t_gm, t_mte1) + prologue
    return t_total, t_cube, t_gm, t_mte1, t_a, t_b


def simulate_core(arch: Arch, hier: Hierarchy, m: int, n: int, k: int) -> CoreResult:
    tile = select_tile(arch, hier, m, n, k)
    s = arch.dtype_in
    if tile is None:
        return CoreResult(
            m, n, k, None, math.inf, 0, 0, 0, 0, 0, "infeasible",
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, hier.label(), "no legal tile",
        )
    t_total, t_cube, t_gm, t_mte1, t_a, t_b = resource_times(arch, hier, m, n, k, tile)
    a_reload, b_reload = reload_factors(arch, hier, tile, m, n)
    bound = max({"cube": t_cube, "gm": t_gm, "mte1": t_mte1}, key=lambda x: {"cube": t_cube, "gm": t_gm, "mte1": t_mte1}[x])
    l1_work = (tile.m_blk * tile.k_l1 + tile.k_l1 * tile.n_blk) * s
    l0a_work = tile.m_l0 * tile.k_l0 * s if hier.has_l0a else 0
    l0b_work = tile.k_l0 * tile.n_l0 * s if hier.has_l0b else 0
    gm_bytes = (m * k + n * k) * s
    return CoreResult(
        m=m, n=n, k=k, tile=tile, cycles=t_total,
        t_cube=t_cube, t_gm=t_gm, t_mte1=t_mte1, t_mte1_a=t_a, t_mte1_b=t_b,
        bound=bound,
        cube_util=t_cube / t_total if t_total else 0.0,
        gm_util=t_gm / t_total if t_total else 0.0,
        a_reload=a_reload, b_reload=b_reload,
        l0a_over_l1=l0a_work / l1_work if l1_work else 0.0,
        l0b_over_l1=l0b_work / l1_work if l1_work else 0.0,
        l1_working_bytes=l1_work,
        l0a_working_bytes=l0a_work,
        l0b_working_bytes=l0b_work,
        intensity=(2 * m * n * k) / gm_bytes if gm_bytes else 0.0,
        hierarchy=hier.label(),
        notes=(
            f"tile m={tile.m_blk} n={tile.n_blk} k1={tile.k_l1} "
            f"m0={tile.m_l0} n0={tile.n_l0} k0={tile.k_l0} stat={tile.stationary}"
        ),
    )


def simulate_gemm(arch: Arch, hier: Hierarchy, m: int, n: int, k: int, block_num: int, split: str) -> CoreResult:
    mc, nc, kc = split_shape(m, n, k, block_num, split)
    r = simulate_core(arch, hier, mc, nc, kc)
    r.notes = f"split={split} blockNum={block_num} per-core ({mc},{nc},{kc}); " + r.notes
    return r


HIER_VARIANTS = (
    Hierarchy(False, False, shared_port=True),
    Hierarchy(False, False, shared_port=False),
    Hierarchy(True, False),
    Hierarchy(False, True),
    Hierarchy(True, True),
)


# ---------------------------------------------------------------------------
# Closed-form proof (no tiler). Source of the markdown equations.
# ---------------------------------------------------------------------------

@dataclass
class Closed:
    name: str
    m: int
    n: int
    k: int
    t_cube: float
    t_gm: float
    t_a: float
    t_b: float
    t_mte1: float
    t: float
    bound: str
    a_reload: float
    b_reload: float
    bw_a: float
    has_l0a: bool
    has_l0b: bool
    s_l0a: int
    s_l0b: int

    @property
    def cube_util(self) -> float:
        return self.t_cube / self.t if self.t else 0.0


def closed_form(
    arch: Arch,
    m: int,
    n: int,
    k: int,
    has_l0a: bool,
    has_l0b: bool,
    bw_a: Optional[float] = None,
    a_from_l1_mmad: bool = False,
    split_k_cstore: bool = False,
) -> Closed:
    """Roofline (7): T = max(T_cube, T_GM, T_MTE1), no prologue.

    a_from_l1_mmad: Cube/MMAD issues A from L1 every 16-wide pulse (reload n/C)
    even if bw_a is boosted. This is the 'cancel L0A, raise this path' model.
    """
    s, p, c = arch.dtype_in, arch.macs_per_cycle, arch.cube_n
    ba = arch.bw_l1_l0a if bw_a is None else bw_a
    bb = arch.bw_l1_l0b
    t_cube = m * n * k / p
    t_gm = (m * k + n * k) * s / arch.bw_gm_l1
    if split_k_cstore:
        t_gm += m * n * s / arch.bw_gm_l1

    if a_from_l1_mmad or not has_l0a:
        a_reload = n / c
    else:
        a_reload = 1.0
    b_reload = 1.0 if has_l0b else (m / arch.cube_m)

    t_a = m * k * s * a_reload / ba
    t_b = n * k * s * b_reload / bb
    if not has_l0b:
        t_b = max(t_b, t_cube * (arch.cube_b_bw / bb))

    shared = (a_from_l1_mmad or not has_l0a) and not has_l0b
    t_mte1 = (t_a + t_b) if shared else max(t_a, t_b)
    t = max(t_cube, t_gm, t_mte1)
    bound = max({"cube": t_cube, "gm": t_gm, "mte1": t_mte1}, key=lambda x: {"cube": t_cube, "gm": t_gm, "mte1": t_mte1}[x])
    k0 = min(k, arch.cube_k)
    s_l0a = 2 * m * k0 * s if has_l0a else 0
    s_l0b = 2 * k0 * n * s if has_l0b else 0
    # clip to physical L0 if the unsplit operand does not fit: tiler would n-panel.
    if has_l0a:
        s_l0a = min(s_l0a, arch.l0a_size)
    if has_l0b:
        s_l0b = min(s_l0b, arch.l0b_size)
    return Closed(
        "", m, n, k, t_cube, t_gm, t_a, t_b, t_mte1, t, bound,
        a_reload, b_reload, ba, has_l0a, has_l0b, s_l0a, s_l0b,
    )


def required_mmad_bw(arch: Arch, n: int) -> float:
    """B_L1,A so that MMAD-from-L1 (reload n/C) matches L0A fill T_A = mKs/B_A.

    Cancel m,K,s: B_L1,A = B_A * (n/C).
    """
    return arch.bw_l1_l0a * (n / arch.cube_n)


def proof_tables(arch: Arch, m: int = 1024, n: int = 1024, k: int = 1024, g: int = 32) -> str:
    lines = ["## Closed-form proof tables\n"]
    specs = [
        ("SplitA", "A", False),
        ("SplitB", "B", False),
        ("SplitK", "K", True),
        ("SplitA+B", "2D", False),
    ]
    cfgs = [
        ("L1-only", False, False, None, False),
        ("L0A-only", True, False, None, False),
        ("L0B-only", False, True, None, False),
        ("L0A+L0B", True, True, None, False),
    ]
    lines.append("### Single-core no-split (1024^3, compute-bound reference)\n")
    lines.append(
        "| hier | A-reload | B-reload | T_cube | T_GM | T_A | T_B | T | bound | vs L1-only |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    base1 = closed_form(arch, m, n, k, False, False)
    for cname, ha, hb, bwa, mmad in cfgs:
        r = closed_form(arch, m, n, k, ha, hb, bw_a=bwa, a_from_l1_mmad=mmad)
        lines.append(
            f"| {cname} | {r.a_reload:.1f} | {r.b_reload:.1f} | {_fmt(r.t_cube,1)} | "
            f"{_fmt(r.t_gm,1)} | {_fmt(r.t_a,1)} | {_fmt(r.t_b,1)} | {_fmt(r.t,1)} | "
            f"{r.bound} | {base1.t/r.t:.2f}x |"
        )
    lines.append("")
    lines.append("### Per-split hierarchy (1024^3, blockNum=32)\n")
    lines.append(
        "| split | hier | m×n×k | A-reload | B-reload | T_cube | T_GM | T_A | T_B | T | bound | vs L1-only | L0A cap | L0B cap |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for sname, mode, kstore in specs:
        mc, nc, kc = split_shape(m, n, k, g, mode)
        base = closed_form(arch, mc, nc, kc, False, False, split_k_cstore=kstore)
        for cname, ha, hb, bwa, mmad in cfgs:
            r = closed_form(arch, mc, nc, kc, ha, hb, bw_a=bwa, a_from_l1_mmad=mmad, split_k_cstore=kstore)
            r.name = f"{sname}/{cname}"
            sp = base.t / r.t if r.t else 0
            lines.append(
                f"| {sname} | {cname} | {mc}×{nc}×{kc} | {r.a_reload:.1f} | {r.b_reload:.1f} | "
                f"{_fmt(r.t_cube,1)} | {_fmt(r.t_gm,1)} | {_fmt(r.t_a,1)} | {_fmt(r.t_b,1)} | "
                f"{_fmt(r.t,1)} | {r.bound} | {sp:.2f}x | {r.s_l0a} | {r.s_l0b} |"
            )
    lines.append("")

    lines.append("### Cancel L0A, MMAD from L1, boost A-path ×μ (keep L0B)\n")
    lines.append(
        "| split | n | μ | B_L1,A | μ* = n/16 | T_A | T | bound | vs L0A+L0B | match? |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for sname, mode, kstore in specs:
        mc, nc, kc = split_shape(m, n, k, g, mode)
        gold = closed_form(arch, mc, nc, kc, True, True, split_k_cstore=kstore)
        mu_star = nc / arch.cube_n
        for mu in (1, 2, 4, 8, 16, 32, 64):
            r = closed_form(
                arch, mc, nc, kc, False, True,
                bw_a=arch.bw_l1_l0a * mu, a_from_l1_mmad=True, split_k_cstore=kstore,
            )
            match = "yes" if r.t <= gold.t * 1.02 else "no"
            lines.append(
                f"| {sname} | {nc} | {mu} | {arch.bw_l1_l0a*mu:.0f} | {mu_star:.0f} | "
                f"{_fmt(r.t_a,1)} | {_fmt(r.t,1)} | {r.bound} | {r.t/gold.t:.2f}×gold | {match} |"
            )
    lines.append("")
    lines.append(
        "Roofline note: T_A(MMAD)/T_cube = P s / (C B_A) = 1 when B_A=512, "
        "independent of n. Boosting μ>1 cannot beat Cube time; it only helps "
        "if L1 A-port was narrower than 512, or if pulse latency / shared ports apply.\n"
    )
    lines.append("### MMAD-from-L1 plus L1 pulse latency L (keep L0B, μ=1, B_A=512)\n")
    lines.append("| split | L (cyc/pulse) | N_pulse | T_A+N L | T | vs L0A+L0B |")
    lines.append("|---|---|---|---|---|---|")
    for sname, mode, kstore in specs:
        mc, nc, kc = split_shape(m, n, k, g, mode)
        gold = closed_form(arch, mc, nc, kc, True, True, split_k_cstore=kstore)
        r = closed_form(arch, mc, nc, kc, False, True, a_from_l1_mmad=True, split_k_cstore=kstore)
        n_pulse = (mc / arch.cube_m) * (nc / arch.cube_n) * (kc / arch.cube_k)
        for lat in (0, 1, 16):
            t_al = r.t_a + n_pulse * lat
            t = max(r.t_cube, r.t_gm, t_al, r.t_b)
            lines.append(
                f"| {sname} | {lat} | {_fmt(n_pulse,1)} | {_fmt(t_al,1)} | {_fmt(t,1)} | {t/gold.t:.2f}x |"
            )
    lines.append("")
    lines.append("### Required μ* and capacity (ping-pong, one K-slice of cube_k=16)\n")
    lines.append("| split | m×n×k | μ*=n/16 | B_L1,A* (B/cyc) | L0A≥2 m C s | L0B≥2 C n s | η_A=n/C | η_B=m/C |")
    lines.append("|---|---|---|---|---|---|---|---|")
    s = arch.dtype_in
    for sname, mode, _kstore in specs:
        mc, nc, kc = split_shape(m, n, k, g, mode)
        mu_star = nc / arch.cube_n
        cap_a = 2 * mc * arch.cube_k * s
        cap_b = 2 * arch.cube_k * nc * s
        lines.append(
            f"| {sname} | {mc}×{nc}×{kc} | {mu_star:.0f} | {arch.bw_l1_l0a*mu_star:.0f} | "
            f"{cap_a} | {cap_b} | {nc/arch.cube_n:.1f} | {mc/arch.cube_m:.1f} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def _fmt(x: float, digits: int = 2) -> str:
    if math.isinf(x):
        return "inf"
    if abs(x) >= 1e6:
        return f"{x/1e6:.{digits}f}M"
    if abs(x) >= 1e3:
        return f"{x/1e3:.{digits}f}k"
    return f"{x:.{digits}f}"


def print_result_table(rows: Sequence[CoreResult], title: str) -> str:
    lines = [f"## {title}", ""]
    lines.append(
        "| hier | per-core MxNxK | bound | cube% | gm% | A-reload | B-reload "
        "| L0A/L1 | L0B/L1 | cycles | vs L1-shared |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    baseline = {}
    for r in rows:
        key = (r.m, r.n, r.k, r.notes.split(";")[0])
        if r.hierarchy == "L1-only/sharedL1":
            baseline[key] = r.cycles
    for r in rows:
        key = (r.m, r.n, r.k, r.notes.split(";")[0])
        base = baseline.get(key, r.cycles)
        sp = base / r.cycles if r.cycles else 0
        lines.append(
            f"| {r.hierarchy} | {r.m}x{r.n}x{r.k} | {r.bound} | "
            f"{100*r.cube_util:.1f}% | {100*r.gm_util:.1f}% | "
            f"{r.a_reload:.1f} | {r.b_reload:.1f} | "
            f"{r.l0a_over_l1:.3f} | {r.l0b_over_l1:.3f} | "
            f"{_fmt(r.cycles, 1)} | {sp:.2f}x |"
        )
    lines.append("")
    return "\n".join(lines)


def _svg_begin(w: int, h: int) -> List[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:DejaVu Sans,Liberation Sans,sans-serif;font-size:12px;fill:#222}</style>',
    ]


def write_line_svg(path, series, xlabel, ylabel, title, width=740, height=430):
    pad_l, pad_r, pad_t, pad_b = 70, 24, 40, 56
    xs = [x for _, xv, _ in series for x in xv]
    ys = [y for _, _, yv in series for y in yv]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = 0.0, (max(ys) * 1.12 if ys else 1.0)
    if xmax <= xmin:
        xmax = xmin + 1
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b

    def sx(x):
        return pad_l + (x - xmin) / (xmax - xmin) * pw

    def sy(y):
        return pad_t + (1 - (y - ymin) / max(ymax - ymin, 1e-9)) * ph

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    out = _svg_begin(width, height)
    out.append(f'<text x="{width/2}" y="24" text-anchor="middle" font-weight="700">{title}</text>')
    out.append(f'<text x="{width/2}" y="{height-12}" text-anchor="middle">{xlabel}</text>')
    out.append(
        f'<text x="16" y="{height/2}" text-anchor="middle" transform="rotate(-90 16 {height/2})">{ylabel}</text>'
    )
    out.append(f'<rect x="{pad_l}" y="{pad_t}" width="{pw}" height="{ph}" fill="none" stroke="#ccc"/>')
    for i in range(5):
        y = ymin + (ymax - ymin) * i / 4
        out.append(f'<line x1="{pad_l}" y1="{sy(y):.1f}" x2="{pad_l+pw}" y2="{sy(y):.1f}" stroke="#eee"/>')
        out.append(f'<text x="{pad_l-8}" y="{sy(y)+4:.1f}" text-anchor="end">{y:.2f}</text>')
    uniq = sorted(set(xs))
    step = max(1, len(uniq) // 8)
    for i, x in enumerate(uniq):
        if i % step != 0 and i != len(uniq) - 1:
            continue
        out.append(f'<text x="{sx(x):.1f}" y="{pad_t+ph+18}" text-anchor="middle">{x:g}</text>')
    for i, (name, xv, yv) in enumerate(series):
        c = colors[i % len(colors)]
        pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xv, yv))
        out.append(f'<polyline fill="none" stroke="{c}" stroke-width="2.2" points="{pts}"/>')
        for x, y in zip(xv, yv):
            out.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3.2" fill="{c}"/>')
        out.append(f'<text x="{pad_l+8}" y="{pad_t+16+i*16}" fill="{c}">{name}</text>')
    out.append("</svg>\n")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out[:-1]) + "\n" + out[-1])


def write_bar_svg(path, labels, groups, ylabel, title, width=780, height=430):
    pad_l, pad_r, pad_t, pad_b = 64, 16, 40, 78
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    n, g = len(labels), len(groups)
    ymax = max((v for _, vals in groups for v in vals), default=1.0) * 1.15 or 1.0
    colors = ["#7f7f7f", "#c7c7c7", "#1f77b4", "#ff7f0e", "#2ca02c"]
    out = _svg_begin(width, height)
    out.append(f'<text x="{width/2}" y="24" text-anchor="middle" font-weight="700">{title}</text>')
    out.append(
        f'<text x="16" y="{height/2}" text-anchor="middle" transform="rotate(-90 16 {height/2})">{ylabel}</text>'
    )
    slot = pw / max(n, 1)
    bar_w = slot / (g + 1.5)
    for i in range(5):
        y = ymax * i / 4
        yy = pad_t + (1 - i / 4) * ph
        out.append(f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{pad_l+pw}" y2="{yy:.1f}" stroke="#eee"/>')
        out.append(f'<text x="{pad_l-8}" y="{yy+4:.1f}" text-anchor="end">{y:.2f}</text>')
    for i, lab in enumerate(labels):
        x0 = pad_l + i * slot + bar_w * 0.6
        out.append(f'<text x="{x0 + g*bar_w/2:.1f}" y="{pad_t+ph+18}" text-anchor="middle">{lab}</text>')
        for j, (_, vals) in enumerate(groups):
            v = vals[i]
            h = v / ymax * ph
            x = x0 + j * bar_w
            y = pad_t + ph - h
            out.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-2:.1f}" height="{h:.1f}" fill="{colors[j%len(colors)]}"/>'
            )
    for j, (name, _) in enumerate(groups):
        out.append(f'<rect x="{pad_l+j*145}" y="{height-32}" width="10" height="10" fill="{colors[j%len(colors)]}"/>')
        out.append(f'<text x="{pad_l+j*145+14}" y="{height-22}">{name}</text>')
    out.append("</svg>\n")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out[:-1]) + "\n" + out[-1])


def write_csv(path: str, rows: Sequence[CoreResult]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = [
        "hierarchy", "m", "n", "k", "bound", "cycles", "cube_util", "gm_util",
        "a_reload", "b_reload", "l0a_over_l1", "l0b_over_l1",
        "l1_working_bytes", "l0a_working_bytes", "l0b_working_bytes",
        "intensity", "notes",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: getattr(r, k) if k != "hierarchy" else r.hierarchy for k in fields})


def analytical_lines(arch: Arch) -> str:
    s = arch.dtype_in
    p = arch.macs_per_cycle
    lines = ["## Closed-form necessities (independent of the tiler)\n"]
    lines.append(
        f"- Cube FP16 peak: {p} MAC/cyc. A-port at n=16: "
        f"`P*s/n = {arch.cube_a_bw:.0f} B/cyc` (matches L1→L0A={arch.bw_l1_l0a:.0f}).\n"
    )
    lines.append(
        f"- B-port at m=16: `{arch.cube_b_bw:.0f} B/cyc`, but L1→L0B="
        f"{arch.bw_l1_l0b:.0f}. A-fill hide `n_l0 >= {arch.min_n_for_l0a_fill:.0f}`; "
        f"B-fill hide `m_l0 >= {arch.min_m_for_l0b_fill:.0f}`.\n"
    )
    lines.append(
        f"- 搬入-bound iff `mn/(m+n) < P*s/B_GM = {arch.compute_bound_pair_size:.0f}`. "
        f"Square GEMM needs m=n>`{2*arch.compute_bound_pair_size:.0f}` to be compute-bound from HBM.\n"
    )
    lines.append(
        "- Without L0A, L1 A-traffic is `m*K*s*(n/16)`; without L0B, L1 B-traffic is `n*K*s*(m/16)`.\n"
    )
    lines.append(
        "- L1 ping-pong working set `2*k_l1*(m+n)*s`; L0A ping-pong `2*m0*k0*s`. "
        "Ratio `L0A/L1 ≈ (m0*k0)/(k_l1*(m+n))`. With typical k_l1=4..8 k0 and m≈n, ratio ≈ 1/8 to 1/16.\n"
    )
    # numeric examples
    for m, n, k, b, split, name in (
        (1024, 1024, 1024, 1, "A", "1-core 1024^3"),
        (1024, 1024, 1024, 32, "A", "32-core A-split"),
        (1024, 1024, 1024, 32, "2D", "32-core 2D-split"),
        (1024, 1024, 1024, 32, "K", "32-core K-split"),
        (16, 4096, 4096, 8, "B", "decode-like 16x4096"),
        (1024, 4096, 1024, 1, "A", "wide-N 1024x4096"),
        (4096, 1024, 1024, 1, "A", "tall-M 4096x1024"),
    ):
        mc, nc, kc = split_shape(m, n, k, b, split)
        pair = mc * nc / (mc + nc)
        a_r = nc / 16
        b_r = mc / 16
        t_cube = mc * nc * kc / p
        t_gm = (mc * kc + nc * kc) * s / arch.bw_gm_l1
        t_a = mc * kc * s * a_r / arch.bw_l1_l0a
        t_b = nc * kc * s * b_r / arch.bw_l1_l0b
        t_nolo = max(t_cube, t_gm, t_a + t_b)
        t_l0 = max(t_cube, t_gm, mc * kc * s / arch.bw_l1_l0a, nc * kc * s / arch.bw_l1_l0b)
        lines.append(
            f"- **{name}**: per-core {mc}x{nc}x{kc}, pair={pair:.1f} "
            f"({'搬入' if pair < arch.compute_bound_pair_size else '计算'}Bound); "
            f"no-L0 T≈{_fmt(t_nolo,1)} (A-reload {a_r:.0f}x, B-reload {b_r:.0f}x) vs "
            f"L0 T≈{_fmt(t_l0,1)} → **{t_nolo/t_l0:.2f}x**.\n"
        )
    return "".join(lines) + "\n"


def self_check(arch: Arch) -> List[str]:
    msgs = []
    big_l0 = simulate_core(arch, Hierarchy(True, True), 1024, 1024, 1024)
    big_no = simulate_core(arch, Hierarchy(False, False, shared_port=True), 1024, 1024, 1024)
    assert big_l0.cycles < big_no.cycles, (big_l0.cycles, big_no.cycles)
    assert big_l0.l1_working_bytes <= arch.l1_size, big_l0.l1_working_bytes
    msgs.append(
        f"OK 1024^3 L0A+L0B {big_l0.cycles:.0f} < L1-shared {big_no.cycles:.0f} "
        f"({big_no.cycles/big_l0.cycles:.2f}x); L1 ws {big_l0.l1_working_bytes}"
    )
    assert 0 < big_l0.cube_util <= 1.0
    mc, nc, _ = split_shape(1024, 1024, 1024, 32, "A")
    assert is_load_bound(mc, nc, arch)
    msgs.append(f"OK A-split 32c per-core {mc}x{nc} is 搬入-bound")
    r8 = simulate_core(arch, Hierarchy(True, True, l0a_size=8 * 1024), 1024, 1024, 1024)
    r64 = simulate_core(arch, Hierarchy(True, True, l0a_size=64 * 1024), 1024, 1024, 1024)
    assert r64.cycles <= r8.cycles * 1.05, (r64.cycles, r8.cycles)
    msgs.append(f"OK L0A 64KB ({r64.cycles:.0f}) <= 8KB ({r8.cycles:.0f})")
    assert abs(arch.cube_a_bw - arch.bw_l1_l0a) < 1e-6
    msgs.append(f"OK Cube A-port {arch.cube_a_bw:.0f} == L1→L0A")
    msgs.append(
        f"OK hide-fill n_l0>={arch.min_n_for_l0a_fill:.0f} (A), "
        f"m_l0>={arch.min_m_for_l0b_fill:.0f} (B)"
    )
    mk, nk, kk = split_shape(1024, 1024, 1024, 32, "K")
    assert kk == 32 and mk == 1024 and nk == 1024
    msgs.append(f"OK SplitK 32c per-core {mk}x{nk}x{kk}")
    gold = closed_form(arch, 1024, 1024, 1024, True, True)
    mmad1 = closed_form(arch, 1024, 1024, 1024, False, True, bw_a=512, a_from_l1_mmad=True)
    mmad64 = closed_form(arch, 1024, 1024, 1024, False, True, bw_a=512 * 64, a_from_l1_mmad=True)
    assert abs(mmad1.t - gold.t) < 1.0 or mmad1.t >= gold.t
    msgs.append(
        f"OK MMAD-from-L1 μ=1 T={mmad1.t:.0f} vs L0A T={gold.t:.0f} "
        f"(A-pipe T_A/T_cube=1 at 512 B/cyc); μ=64 T={mmad64.t:.0f}"
    )
    only_a = simulate_core(arch, Hierarchy(True, False), 1024, 1024, 1024)
    only_b = simulate_core(arch, Hierarchy(False, True), 1024, 1024, 1024)
    msgs.append(
        f"OK ablation 1024^3 L0A-only {only_a.cycles:.0f} bound={only_a.bound}; "
        f"L0B-only {only_b.cycles:.0f} bound={only_b.bound}"
    )
    wide = simulate_core(arch, Hierarchy(True, True), 1024, 4096, 1024)
    wide_b = simulate_core(arch, Hierarchy(False, True), 1024, 4096, 1024)
    msgs.append(
        f"OK wide-N 1024x4096x1024 L0A+L0B {wide.cycles:.0f} vs L0B-only {wide_b.cycles:.0f} "
        f"({wide_b.cycles / wide.cycles:.2f}x L0A extra)"
    )
    return msgs


def run(out_dir: str, artifact_dir: Optional[str] = None) -> str:
    arch = Arch()
    os.makedirs(out_dir, exist_ok=True)
    md: List[str] = [f"# L0A/L0B software-model run ({arch.name})\n", "Self-check:\n"]
    for msg in self_check(arch):
        md.append(f"- {msg}")
    md.append("")
    md.append(analytical_lines(arch))
    md.append(proof_tables(arch))

    shapes = [
        (1024, 1024, 1024, 1, "A"),
        (1024, 1024, 1024, 32, "A"),
        (1024, 1024, 1024, 32, "B"),
        (1024, 1024, 1024, 32, "2D"),
        (1024, 1024, 1024, 32, "K"),
        (16, 4096, 4096, 8, "B"),
        (2048, 4096, 4096, 24, "A"),
        (4096, 4096, 1024, 32, "2D"),
        (1024, 4096, 1024, 1, "A"),
        (4096, 1024, 1024, 1, "A"),
    ]
    rows = []
    for shape in shapes:
        for hier in HIER_VARIANTS:
            rows.append(simulate_gemm(arch, hier, *shape))
    write_csv(os.path.join(out_dir, "hierarchy_ablation.csv"), rows)
    md.append(print_result_table(rows, "Hierarchy ablation"))

    sizes = [4, 8, 16, 32, 64, 128, 256]

    def ratio_sweep(which: str, block: int, split: str):
        out = []
        for kb in sizes:
            size = kb * 1024
            if which == "A":
                hier = Hierarchy(True, True, l0a_size=size)
            elif which == "B":
                hier = Hierarchy(True, True, l0b_size=size)
            else:
                hier = Hierarchy(True, True, l0a_size=size, l0b_size=size)
            r = simulate_gemm(arch, hier, 1024, 1024, 1024, block, split)
            out.append((kb, size / arch.l1_size, r))
        return out

    ra32 = ratio_sweep("A", 32, "A")
    rb32 = ratio_sweep("B", 32, "A")
    ra1 = ratio_sweep("A", 1, "A")
    rb1 = ratio_sweep("B", 1, "A")
    rab1 = ratio_sweep("AB", 1, "A")

    md.append("## L0 vs L1 capacity (L1=512KB fixed, 1024^3)\n")
    md.append("| L0 KB | L0/L1 | 32c A-split L0A cube% | 32c L0B cube% | 1c L0A cube% | 1c L0B cube% | 1c both cube% |")
    md.append("|---|---|---|---|---|---|---|")
    for i, kb in enumerate(sizes):
        md.append(
            f"| {kb} | {kb*1024/arch.l1_size:.3f} | "
            f"{100*ra32[i][2].cube_util:.1f}% | {100*rb32[i][2].cube_util:.1f}% | "
            f"{100*ra1[i][2].cube_util:.1f}% | {100*rb1[i][2].cube_util:.1f}% | "
            f"{100*rab1[i][2].cube_util:.1f}% |"
        )
    md.append("")

    def sp(items):
        base = items[0][2].cycles
        return [base / it[2].cycles for it in items]

    write_line_svg(
        os.path.join(out_dir, "l0_ratio_speedup.svg"),
        [
            ("L0A sweep, 32c A-split 搬入Bound", [r[1] for r in ra32], sp(ra32)),
            ("L0B sweep, 32c A-split 搬入Bound", [r[1] for r in rb32], sp(rb32)),
            ("L0A sweep, 1-core", [r[1] for r in ra1], sp(ra1)),
            ("L0B sweep, 1-core", [r[1] for r in rb1], sp(rb1)),
            ("L0A+L0B sweep, 1-core", [r[1] for r in rab1], sp(rab1)),
        ],
        xlabel="L0 / L1 capacity ratio",
        ylabel="Speedup vs 4KB L0",
        title="L0/L1 ratio: compute-bound saturates by 8KB; 64KB=1/8 is spec margin",
    )

    blocks = [1, 2, 4, 8, 16, 24, 32]

    def bn_sweep(split):
        rows_bn = []
        for b in blocks:
            yes = simulate_gemm(arch, Hierarchy(True, True), 1024, 1024, 1024, b, split)
            no = simulate_gemm(arch, Hierarchy(False, False, shared_port=True), 1024, 1024, 1024, b, split)
            dual = simulate_gemm(arch, Hierarchy(False, False, shared_port=False), 1024, 1024, 1024, b, split)
            rows_bn.append((b, yes, no, dual))
        return rows_bn

    bn_a = bn_sweep("A")
    bn_2d = bn_sweep("2D")
    write_line_svg(
        os.path.join(out_dir, "blocknum_speedup.svg"),
        [
            ("A-split vs L1-shared", [t[0] for t in bn_a], [t[2].cycles / t[1].cycles for t in bn_a]),
            ("A-split vs L1-dualR", [t[0] for t in bn_a], [t[3].cycles / t[1].cycles for t in bn_a]),
            ("2D-split vs L1-shared", [t[0] for t in bn_2d], [t[2].cycles / t[1].cycles for t in bn_2d]),
            ("2D-split vs L1-dualR", [t[0] for t in bn_2d], [t[3].cycles / t[1].cycles for t in bn_2d]),
        ],
        xlabel="blockNum",
        ylabel="Speedup of L0A+L0B over L1-only",
        title="Multi-core split: L0 gain vs reload/port contention",
    )

    labels = ["1c 1024^3", "32c A-split", "32c 2D-split", "32c K-split", "8c 16x4096"]
    scenarios = [
        (1024, 1024, 1024, 1, "A"),
        (1024, 1024, 1024, 32, "A"),
        (1024, 1024, 1024, 32, "2D"),
        (1024, 1024, 1024, 32, "K"),
        (16, 4096, 4096, 8, "B"),
    ]
    groups = []
    for hier in HIER_VARIANTS:
        groups.append((hier.label(), [simulate_gemm(arch, hier, *sc).cube_util for sc in scenarios]))
    write_bar_svg(
        os.path.join(out_dir, "ablation_cube_util.svg"),
        labels, groups, ylabel="Cube utilization",
        title="Cube util with/without L0A and L0B",
    )

    md.append("## blockNum sweep (1024^3)\n")
    md.append(
        "| blockNum | split | pair | 搬入Bound | L0 | L1-shared | L1-dualR | "
        "vs shared | vs dualR | L0 bound | shared bound |"
    )
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for split, series in (("A", bn_a), ("2D", bn_2d)):
        for b, yes, no, dual in series:
            pair = yes.m * yes.n / (yes.m + yes.n)
            md.append(
                f"| {b} | {split} | {pair:.1f} | {'yes' if is_load_bound(yes.m, yes.n, arch) else 'no'} | "
                f"{_fmt(yes.cycles,1)} | {_fmt(no.cycles,1)} | {_fmt(dual.cycles,1)} | "
                f"{no.cycles/yes.cycles:.2f}x | {dual.cycles/yes.cycles:.2f}x | "
                f"{yes.bound} | {no.bound} |"
            )
    md.append("")

    md.append("## Default 64KB L0 working-set vs L1 panel\n")
    md.append("| scenario | L1 ws | L0A ws | L0B ws | L0A/L1 | L0B/L1 | (L0A+L0B)/L1 | tile |")
    md.append("|---|---|---|---|---|---|---|---|")
    for sc in scenarios:
        r = simulate_gemm(arch, Hierarchy(True, True), *sc)
        md.append(
            f"| {sc} | {r.l1_working_bytes} | {r.l0a_working_bytes} | {r.l0b_working_bytes} | "
            f"{r.l0a_over_l1:.3f} | {r.l0b_over_l1:.3f} | "
            f"{(r.l0a_over_l1+r.l0b_over_l1):.3f} | {r.notes} |"
        )
    md.append("")

    report = "\n".join(md) + "\n"
    with open(os.path.join(out_dir, "run_report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    if artifact_dir:
        os.makedirs(artifact_dir, exist_ok=True)
        for name in os.listdir(out_dir):
            src = os.path.join(out_dir, name)
            if os.path.isfile(src):
                with open(src, "rb") as fi, open(os.path.join(artifact_dir, name), "wb") as fo:
                    fo.write(fi.read())
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "out"))
    p.add_argument("--artifacts", default="")
    args = p.parse_args()
    print(run(args.out, args.artifacts or None))


if __name__ == "__main__":
    main()
