# Collective operator DSL

RankTable → 拓扑模型 → 算法模板 → 算子实例。方案见 [docs/hcomm/02-coll-op-topo-alg-dsl.md](../../docs/hcomm/02-coll-op-topo-alg-dsl.md)。

## 命令

```bash
python3 coll_op_dsl.py topo  ../ranktable_dsl/examples/v2_two_layer.rt
python3 coll_op_dsl.py lower examples/allreduce_hier.coll --rank 0
python3 coll_op_dsl.py sim   examples/allreduce_ring.coll -v
python3 verify.py
python3 -m unittest tests/test_coll_op_dsl.py
```

| 命令 | 作用 |
|------|------|
| `topo` | RankTable DSL/JSON → 拓扑模型（layer / instance / ranks） |
| `lower` | 算子 + 算法模板 → send/recv/reduce schedule |
| `sim` | 模拟执行并核对集合语义 |
| `verify.py` | 同一 AllReduce、两套算法结果相同而 schedule 不同 |

`examples/*.coll` 里 `topo from` 指向 `ranktable_dsl/examples` 中的表。
