# 常用网络算子调用剖析

对典型 CNN / Transformer / 检测 / 分割 / RNN 做一次 **ATen 叶子调用次数 + 排他 CPU 耗时 + Cube FLOPs** 统计。

```bash
python tools/net_op_profile/profile_common_nets.py
python tools/net_op_profile/profile_common_nets.py --only ResNet-50 BERT-Base
```

结果写到 `results/summary.json` 与 `results/tables.md`。需要 PyTorch + torchvision（CPU 即可）。

Pattern 编号对齐 [pytorch_op_feature_compete.md](../../docs/design/pytorch_op_feature_compete.md)。
