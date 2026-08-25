# code

CCU / HCOMM 相关分析笔记与工具。

## 文档

- [度量框架审计](docs/metrics-framework/README.md)（G0–G3 对照；现行 G1 权重表不可按公式复现）
- [HCOMM CCU 底层指令格式分析](docs/ccu/01-underlying-instruction-format.md)
- [PyTorch 算子特征分类、API 特征与同步特征提取](docs/design/pytorch_op_feature_compete.md)
- [CUDA L2 Cache Persistence 控制机制](docs/cuda/a.md)
- [下一代调试调优度量体系](docs/下一代调试调优度量体系.md)（G1 正文）
- [硬件易用性度量（G3 / next_me）](docs/next_me.md)（从 PR #13 `docs/next.md` 检出，本仓库无独立 `next_me.md` 原稿）

## 工具

- [CCU V1 汇编器](tools/ccu_v1_asm/README.md)
  - C 实现：`tools/ccu_v1_asm/c`
  - C 风格：`create ctx → call main() → 指令填二进制 → write_file`
  - 位置操作数、变量分配/复用、metainfo
