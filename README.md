# code

CCU / HCOMM 相关分析笔记与工具。

## 文档

- [HCOMM CCU 底层指令格式分析](docs/ccu/01-underlying-instruction-format.md)
- [PyTorch 算子特征分类、API 特征与同步特征提取](docs/design/pytorch_op_feature_compete.md)
- [CUDA L2 Cache Persistence 控制机制](docs/cuda/a.md)
- [下一代调试调优度量体系（三档性能 + 六类负载亲和）](docs/下一代调试调优度量体系.md)
  - 目录接口：分类 → 度量项 → 度量子项（文首）
  - 机读目录：[docs/metrics/catalog.yaml](docs/metrics/catalog.yaml)

## 工具

- [CCU V1 汇编器](tools/ccu_v1_asm/README.md)
  - C 实现：`tools/ccu_v1_asm/c`
  - C 风格：`create ctx → call main() → 指令填二进制 → write_file`
  - 位置操作数、变量分配/复用、metainfo
