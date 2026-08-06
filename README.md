# code

## Ascend 950 多 Pipeline 分析

详见 [docs/ascend950-multi-pipeline-analysis.md](docs/ascend950-multi-pipeline-analysis.md)。

基于昇腾社区与 asc-devkit，覆盖：

1. 编程模型（TPipe/TQue 与 3510 扩展）
2. 多 Pipeline 与同步分层
3. SetFlag/WaitFlag 与 BufferId（EventID / MutexID）同步
4. 多 Pipeline 演进与易用性
5. 对比 GPU：多流水在性能发挥与硬件设计上的好处
