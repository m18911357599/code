#!/usr/bin/env python3
"""Profile common networks: ATen call counts, exclusive CPU time, estimated FLOPs."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode

from models import Workload, build_workloads
from pattern_map import classify, normalize_op_name


def _numel(x: Any) -> int:
    if torch.is_tensor(x):
        return int(x.numel())
    return 0


def _shape(x: Any) -> tuple[int, ...] | None:
    if torch.is_tensor(x):
        return tuple(int(d) for d in x.shape)
    return None


def estimate_flops(op_key: str, args: tuple, kwargs: dict, out: Any) -> int:
    """Rough FLOPs (MAC counted as 2). Wrappers may also match; caller keeps leaf only."""
    k = op_key
    try:
        if "convolution" in k or k in {"conv2d", "conv1d", "conv3d", "conv_transpose2d"}:
            inp = args[0]
            w = args[1]
            if not (torch.is_tensor(inp) and torch.is_tensor(w)):
                return 0
            n, cin, *spatial_in = inp.shape
            cout, cg, *ksp = w.shape
            groups = int(args[8]) if len(args) > 8 else int(kwargs.get("groups", 1))
            # output spatial from result if tensor
            if torch.is_tensor(out):
                spatial_out = out.shape[2:]
            else:
                spatial_out = spatial_in
            spat = 1
            for d in spatial_out:
                spat *= int(d)
            kvol = 1
            for d in ksp:
                kvol *= int(d)
            # groups: cin_per_group = w.shape[1] = cg
            return int(n) * int(cout) * spat * int(cg) * kvol * 2

        if k in {"mm"}:
            a, b = args[0], args[1]
            m, n_ = a.shape[0], b.shape[1]
            kk = a.shape[1]
            return int(m * n_ * kk * 2)
        if k in {"addmm"}:
            # addmm(bias, mat1, mat2): out = beta*bias + alpha*mat1@mat2
            mat1, mat2 = args[1], args[2]
            m, kk = mat1.shape
            n_ = mat2.shape[1]
            return int(m * n_ * kk * 2 + m * n_)
        if k in {"bmm"}:
            a, b = args[0], args[1]
            bs, m, kk = a.shape
            n_ = b.shape[-1]
            return int(bs * m * n_ * kk * 2)
        if k in {"matmul"}:
            a, b = args[0], args[1]
            if a.dim() == 2 and b.dim() == 2:
                return int(a.shape[0] * a.shape[1] * b.shape[1] * 2)
            # batched
            *batch, m, kk = a.shape
            n_ = b.shape[-1]
            bs = 1
            for d in batch:
                bs *= int(d)
            return int(bs * m * n_ * kk * 2)
        if "scaled_dot_product" in k:
            q = args[0]
            # QK^T + PV ≈ 2 * B * H * S_q * S_k * D
            if q.dim() == 4:
                b, h, sq, d = q.shape
                sk = args[1].shape[-2]
                return int(b * h * sq * sk * d * 4)
            return 0
        if k in {"linear"}:
            x, w = args[0], args[1]
            *batch, in_f = x.shape
            out_f, _ = w.shape
            bs = 1
            for d in batch:
                bs *= int(d)
            return int(bs * in_f * out_f * 2)
        if k in {"_native_multi_head_attention", "multi_head_attention"}:
            q = args[0]
            d_model = int(args[3])
            n_head = int(args[4])
            w_qkv = args[5]
            w_out = args[7]
            b, s, _ = q.shape
            dk = d_model // n_head
            fl = b * s * w_qkv.shape[0] * w_qkv.shape[1] * 2
            fl += b * s * w_out.shape[0] * w_out.shape[1] * 2
            fl += b * n_head * s * s * dk * 4
            return int(fl)
        if k in {"lstm"}:
            x = args[0]
            weights = args[2]
            b, s = (x.shape[0], x.shape[1]) if x.dim() == 3 else (x.shape[1], x.shape[0])
            fl = 0
            for w in weights:
                if torch.is_tensor(w) and w.dim() == 2:
                    fl += int(b * s * w.shape[0] * w.shape[1] * 2)
            return fl
        if k in {"add", "sub", "mul", "div", "relu", "gelu", "silu", "sigmoid", "tanh", "neg", "exp"}:
            t = out if torch.is_tensor(out) else args[0]
            return _numel(t)
        if "layer_norm" in k or "batch_norm" in k:
            t = args[0]
            return _numel(t) * 5
        if "softmax" in k:
            t = args[0]
            return _numel(t) * 8
    except Exception:
        return 0
    return 0


@dataclass
class OpAcc:
    calls: int = 0
    leaf_calls: int = 0
    self_ns: int = 0
    flops: int = 0


@dataclass
class Frame:
    key: str
    pattern: str
    child_patterns: set[str] = field(default_factory=set)
    flops: int = 0
    self_ns: int = 0


class OpStatMode(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.ops: dict[str, OpAcc] = defaultdict(OpAcc)
        self._stack: list[Frame] = []
        self._child_ns = [0]

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # type: ignore[override]
        kwargs = kwargs or {}
        key = str(getattr(func, "_opname", None) or getattr(func, "__name__", "") or func)
        if "." in key:
            key = key.split(".")[0]
        pid, _, _ = classify(key)
        frame = Frame(key=key, pattern=pid)
        self._stack.append(frame)
        self._child_ns.append(0)
        t0 = time.perf_counter_ns()
        out = func(*args, **kwargs)
        total = time.perf_counter_ns() - t0
        children = self._child_ns.pop()
        self_ns = max(0, total - children)
        self._child_ns[-1] += total
        frame.self_ns = self_ns
        frame.flops = estimate_flops(key, args, kwargs, out)
        self._stack.pop()
        if self._stack:
            self._stack[-1].child_patterns.add(pid)

        acc = self.ops[key]
        acc.calls += 1
        acc.self_ns += self_ns
        is_leaf = pid not in frame.child_patterns
        if is_leaf:
            acc.leaf_calls += 1
            acc.flops += frame.flops
        return out


def _call_model(model: nn.Module, inp: object) -> None:
    if isinstance(inp, tuple):
        model(*inp)
    else:
        model(inp)


def profile_one(wl: Workload, warmup: int = 1) -> dict[str, Any]:
    torch.manual_seed(0)
    model = wl.make_model()
    model.eval()
    inp = wl.make_input()
    if isinstance(inp, tuple):
        inp = tuple(t if torch.is_tensor(t) else t for t in inp)

    with torch.inference_mode():
        for _ in range(warmup):
            _call_model(model, inp)

        mode = OpStatMode()
        t0 = time.perf_counter()
        with mode:
            _call_model(model, inp)
        wall_ms = (time.perf_counter() - t0) * 1000.0

    ops = []
    for name, acc in sorted(mode.ops.items(), key=lambda kv: -kv[1].self_ns):
        pid, pname, klass = classify(name)
        ops.append(
            {
                "op": name,
                "pattern": pid,
                "pattern_name": pname,
                "class": klass,
                "calls": acc.calls,
                "leaf_calls": acc.leaf_calls,
                "self_ms": acc.self_ns / 1e6,
                "flops": acc.flops,
            }
        )

    total_self_ms = sum(o["self_ms"] for o in ops)
    total_leaf_calls = sum(o["leaf_calls"] for o in ops)
    total_flops = sum(o["flops"] for o in ops)

    patterns: dict[str, dict[str, Any]] = {}
    for o in ops:
        p = patterns.setdefault(
            o["pattern"],
            {
                "pattern": o["pattern"],
                "pattern_name": o["pattern_name"],
                "class": o["class"],
                "leaf_calls": 0,
                "calls": 0,
                "self_ms": 0.0,
                "flops": 0,
            },
        )
        p["leaf_calls"] += o["leaf_calls"]
        p["calls"] += o["calls"]
        p["self_ms"] += o["self_ms"]
        p["flops"] += o["flops"]

    patt_list = sorted(patterns.values(), key=lambda x: -x["self_ms"])
    for p in patt_list:
        p["time_share_pct"] = 100.0 * p["self_ms"] / total_self_ms if total_self_ms else 0.0
        p["flops_share_pct"] = 100.0 * p["flops"] / total_flops if total_flops else 0.0
        p["call_share_pct"] = 100.0 * p["leaf_calls"] / total_leaf_calls if total_leaf_calls else 0.0

    cube_ms = sum(p["self_ms"] for p in patt_list if p["class"] == "Cube")
    cube_flops = sum(p["flops"] for p in patt_list if p["class"] == "Cube")
    cube_calls = sum(p["leaf_calls"] for p in patt_list if p["class"] == "Cube")

    return {
        "name": wl.name,
        "family": wl.family,
        "description": wl.description,
        "notes": wl.notes,
        "wall_ms": wall_ms,
        "self_ms": total_self_ms,
        "leaf_calls": total_leaf_calls,
        "aten_kinds": len(ops),
        "flops": total_flops,
        "cube_time_share_pct": 100.0 * cube_ms / total_self_ms if total_self_ms else 0.0,
        "cube_flops_share_pct": 100.0 * cube_flops / total_flops if total_flops else 0.0,
        "cube_call_share_pct": 100.0 * cube_calls / total_leaf_calls if total_leaf_calls else 0.0,
        "patterns": patt_list,
        "ops": ops,
    }


def _pct(x: float) -> str:
    return f"{x:.1f}%"


def _ms(x: float) -> str:
    if x >= 100:
        return f"{x:.0f}"
    if x >= 10:
        return f"{x:.1f}"
    if x >= 1:
        return f"{x:.2f}"
    return f"{x:.3f}"


def render_markdown(results: list[dict[str, Any]], meta: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("## 实测环境")
    lines.append("")
    lines.append(f"- PyTorch `{meta['torch']}` / torchvision `{meta['torchvision']}` / 设备 `{meta['device']}`")
    lines.append(f"- 线程数 `{meta['threads']}`；`inference_mode` + `eval`；随机权重（结构与量级与预训练一致）")
    lines.append("- **调用次数**取 ATen 叶子（同 Pattern 子调用不重复计）；**耗时**为排他 CPU self-time")
    lines.append("- **FLOPs** 对卷积/GEMM/SDPA 按 MAC×2 估算，仅计叶子，避免 `conv2d→convolution→mkldnn` 三计")
    lines.append("- CPU 时间份额**不能**直接当 GPU/NPU 份额；FLOPs 与调用次数跨设备更稳")
    lines.append("")

    lines.append("## 跨网络总表")
    lines.append("")
    lines.append(
        "| 网络 | 族 | 叶子调用 | ATen 种类 | 墙钟 ms | Cube 次数% | Cube 耗时% | Cube FLOPs% |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in results:
        lines.append(
            f"| {r['name']} | {r['family']} | {r['leaf_calls']} | {r['aten_kinds']} | "
            f"{_ms(r['wall_ms'])} | {_pct(r['cube_call_share_pct'])} | "
            f"{_pct(r['cube_time_share_pct'])} | {_pct(r['cube_flops_share_pct'])} |"
        )
    lines.append("")

    # Pattern matrix: time share
    all_p = sorted({p["pattern"] for r in results for p in r["patterns"]})
    lines.append("## Pattern 耗时份额矩阵（行=网络，列=Pattern，单位 %）")
    lines.append("")
    header = "| 网络 | " + " | ".join(all_p) + " |"
    sep = "|---|" + "|".join(["---:" for _ in all_p]) + "|"
    lines.append(header)
    lines.append(sep)
    for r in results:
        mp = {p["pattern"]: p["time_share_pct"] for p in r["patterns"]}
        cells = " | ".join(_pct(mp.get(p, 0.0)) if mp.get(p, 0.0) >= 0.05 else "—" for p in all_p)
        lines.append(f"| {r['name']} | {cells} |")
    lines.append("")

    lines.append("## Pattern 叶子调用次数矩阵")
    lines.append("")
    lines.append(header.replace("网络 | ", "网络 | ") if False else "| 网络 | " + " | ".join(all_p) + " |")
    lines.append(sep)
    for r in results:
        mp = {p["pattern"]: p["leaf_calls"] for p in r["patterns"]}
        cells = " | ".join(str(mp[p]) if mp.get(p, 0) else "—" for p in all_p)
        lines.append(f"| {r['name']} | {cells} |")
    lines.append("")

    lines.append("## 各网络 Top 算子（按 self-time）")
    lines.append("")
    for r in results:
        lines.append(f"### {r['name']}")
        lines.append("")
        lines.append(f"{r['description']} {r['notes']}")
        lines.append("")
        lines.append(
            f"墙钟 `{_ms(r['wall_ms'])} ms`，叶子调用 `{r['leaf_calls']}`，"
            f"估算 FLOPs `{r['flops']:.3e}`。"
        )
        lines.append("")
        lines.append("| Pattern | 名称 | 叶子次数 | 次数% | self ms | 耗时% | FLOPs% |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for p in r["patterns"]:
            if p["time_share_pct"] < 0.3 and p["leaf_calls"] < 3:
                continue
            lines.append(
                f"| {p['pattern']} | {p['pattern_name']} | {p['leaf_calls']} | "
                f"{_pct(p['call_share_pct'])} | {_ms(p['self_ms'])} | "
                f"{_pct(p['time_share_pct'])} | {_pct(p['flops_share_pct'])} |"
            )
        lines.append("")
        lines.append("| ATen 算子 | Pattern | 叶子次数 | 包装次数 | self ms | 耗时% |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for o in r["ops"][:18]:
            share = 100.0 * o["self_ms"] / r["self_ms"] if r["self_ms"] else 0.0
            if share < 0.2 and o["leaf_calls"] < 2:
                continue
            lines.append(
                f"| `{o['op']}` | {o['pattern']} | {o['leaf_calls']} | {o['calls']} | "
                f"{_ms(o['self_ms'])} | {_pct(share)} |"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None, help="subset of workload names")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()

    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    workloads = build_workloads()
    if args.only:
        want = {n.lower() for n in args.only}
        workloads = [w for w in workloads if w.name.lower() in want]

    results = []
    for wl in workloads:
        print(f"[profile] {wl.name} ...", flush=True)
        r = profile_one(wl)
        print(
            f"  wall={r['wall_ms']:.1f}ms leaf_calls={r['leaf_calls']} "
            f"cube_time={r['cube_time_share_pct']:.1f}% cube_flops={r['cube_flops_share_pct']:.1f}%",
            flush=True,
        )
        results.append(r)

    args.out.mkdir(parents=True, exist_ok=True)
    meta = {
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "device": "cpu",
        "threads": torch.get_num_threads(),
    }
    payload = {"meta": meta, "results": results}
    json_path = args.out / "summary.json"
    md_path = args.out / "tables.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(results, meta), encoding="utf-8")
    print(f"wrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
