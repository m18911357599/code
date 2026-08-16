# code

CCU / HCOMM 相关分析笔记与工具。

## 文档

- [HCOMM CCU 底层指令格式分析](docs/ccu/01-underlying-instruction-format.md)
- [PyTorch 算子特征分类、API 特征与同步特征提取](docs/design/pytorch_op_feature_compete.md)
- [CUDA L2 Cache Persistence 控制机制](docs/cuda/a.md)
- [硬件易用性度量](docs/硬件易用性度量.md)（七章：系统属性 / 编译友好 / 内存模型 / 计算模型 / 控制模型 / 访存易用性 / 调试调优）
  - 旧稿入口：[下一代调试调优度量体系](docs/下一代调试调优度量体系.md)（已并入上文）

## 工具

- [CCU V1 汇编器](tools/ccu_v1_asm/README.md)
  - C 实现：`tools/ccu_v1_asm/c`
  - C 风格：`create ctx → call main() → 指令填二进制 → write_file`
  - 位置操作数、变量分配/复用、metainfo
