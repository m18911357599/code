# Rank Table DSL

把 [cann/hcomm](https://gitcode.com/cann/hcomm) rank table JSON 收成块结构文本，再编译回 JSON。

结构说明见 [docs/hcomm/01-rank-table-structure.md](../../docs/hcomm/01-rank-table-structure.md)。

## 命令

```bash
python3 ranktable_dsl.py compile  examples/v2_single_layer.rt -o out.json
python3 ranktable_dsl.py decompile fixtures/rank_table_v2.json -o out.rt
python3 ranktable_dsl.py check    examples/v2_single_layer.rt
python3 ranktable_dsl.py dump     examples/v2_two_layer.rt
```

| 命令 | 作用 |
|------|------|
| `compile` | DSL → JSON（按 `version` 选 V1 `server_list` 或 V2 `rank_list`） |
| `decompile` | JSON → DSL |
| `check` | 解析并跑与 hcomm `RankTableInfo::Check` 对齐的约束 |
| `dump` | 打印规范化 DSL |

```bash
python3 -m unittest tests/test_ranktable_dsl.py
```
