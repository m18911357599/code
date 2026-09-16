# Cube L0A / L0B vs L1 software model

Tile-level model for Ascend 910B1 Matmul: multi-core split, GM→L1 `m*ka` / `kb*n` panels, L0 reload, asymmetric L1→L0 bandwidth.

```bash
python3 tools/l0_model/matmul_hierarchy_model.py --out tools/l0_model/out
```

Writes `run_report.md`, `hierarchy_ablation.csv`, and SVG plots. Analysis: [docs/ascend/l0a_l0b_necessity.md](../../docs/ascend/l0a_l0b_necessity.md).
