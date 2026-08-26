"""Topology model derived from a RankTable (RankGraph language face)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import sys

_RT_DIR = Path(__file__).resolve().parent.parent / "ranktable_dsl"
if str(_RT_DIR) not in sys.path:
    sys.path.insert(0, str(_RT_DIR))

import ranktable_dsl as rt  # noqa: E402


@dataclass
class Instance:
    layer: int
    inst_id: str
    net_type: str
    ranks: list[int]


@dataclass
class TopoModel:
    rank_ids: list[int]
    local_id: dict[int, int]
    device_id: dict[int, int]
    layers: dict[int, list[Instance]]

    def layer_ids(self) -> list[int]:
        return sorted(self.layers)

    def instances(self, layer: int) -> list[Instance]:
        return list(self.layers.get(layer, []))

    def instance_of(self, layer: int, rank: int) -> Optional[Instance]:
        for inst in self.instances(layer):
            if rank in inst.ranks:
                return inst
        return None

    def ranks_in(self, layer: int, inst_id: str) -> list[int]:
        for inst in self.instances(layer):
            if inst.inst_id == inst_id:
                return list(inst.ranks)
        return []

    def slot_groups(self, layer: int) -> list[list[int]]:
        """Transpose: ranks sharing the same index inside each instance."""
        insts = self.instances(layer)
        if not insts:
            return []
        width = len(insts[0].ranks)
        if any(len(i.ranks) != width for i in insts):
            raise rt.CheckError(f"layer {layer} instances are not equal width")
        return [[inst.ranks[slot] for inst in insts] for slot in range(width)]

    def describe(self) -> str:
        lines = [f"topo {{", f"  rank_count {len(self.rank_ids)}"]
        for layer in self.layer_ids():
            lines.append(f"  layer {layer} {{")
            for inst in self.instances(layer):
                ranks = ", ".join(str(r) for r in inst.ranks)
                lines.append(
                    f'    instance "{inst.inst_id}" {inst.net_type} ranks [{ranks}]'
                )
            lines.append("  }")
        lines.append("}")
        return "\n".join(lines) + "\n"


def _add_rank(
    buckets: dict[tuple[int, str], list[int]],
    types: dict[tuple[int, str], str],
    layer: int,
    inst_id: str,
    net_type: str,
    rank_id: int,
) -> None:
    key = (layer, inst_id)
    buckets[key].append(rank_id)
    types[key] = net_type


def from_ranktable(table: rt.RankTable) -> TopoModel:
    rt.check(table)
    buckets: dict[tuple[int, str], list[int]] = defaultdict(list)
    types: dict[tuple[int, str], str] = {}
    local_id: dict[int, int] = {}
    device_id: dict[int, int] = {}

    if table.ranks:
        rank_ids = sorted(r.rank_id for r in table.ranks)
        for rank in table.ranks:
            local_id[rank.rank_id] = rank.local_id
            device_id[rank.rank_id] = rank.device_id
            for level in rank.levels:
                _add_rank(
                    buckets, types, level.net_layer, level.net_instance_id, level.net_type, rank.rank_id
                )
    else:
        rank_ids = []
        server_of: dict[int, str] = {}
        for server in table.servers:
            for dev in server.devices:
                rank_ids.append(dev.rank_id)
                local_id[dev.rank_id] = dev.device_id
                device_id[dev.rank_id] = dev.device_id
                server_of[dev.rank_id] = server.server_id
                _add_rank(buckets, types, 0, server.server_id, "TOPO_FILE_DESC", dev.rank_id)
        rank_ids.sort()
        pod_of_server: dict[str, str] = {}
        for pod in table.super_pods:
            for sid in pod.server_ids:
                pod_of_server[sid] = pod.super_pod_id
        if pod_of_server:
            for rid in rank_ids:
                _add_rank(
                    buckets, types, 1, pod_of_server[server_of[rid]], "CLOS", rid
                )
        elif len(table.servers) > 1:
            for rid in rank_ids:
                _add_rank(buckets, types, 1, "world", "CLOS", rid)

    layers: dict[int, list[Instance]] = defaultdict(list)
    for (layer, inst_id), ranks in sorted(buckets.items()):
        inst = Instance(layer, inst_id, types[(layer, inst_id)], sorted(set(ranks)))
        layers[layer].append(inst)
    for layer in layers:
        layers[layer].sort(key=lambda i: i.inst_id)

    return TopoModel(rank_ids, local_id, device_id, dict(layers))


def load_topo(path: str | Path) -> TopoModel:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json" or text.lstrip().startswith("{"):
        import json

        table = rt.load_json(json.loads(text))
    else:
        table = rt.parse_dsl(text)
    return from_ranktable(table)
