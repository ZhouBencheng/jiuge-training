# 验证记录

实验使用四张 32 GiB BI-V150、CoreX 4.4 和工程固定的 Megatron 版本。每份运行报告保留对应的配置、源码和数据记录，便于按同一条件复现。

## 覆盖范围

| 项目 | 验证状态 |
|---|---|
| 环境与资产 | 四卡枚举、NCCL 通信、三套模型下载与索引、中文数据及预处理校验通过 |
| 训练基线验收 | 30 项 unittest 全部通过，耗时 63.30 秒 |
| 并行编排 | 六种四卡 tiny CPT 布局、代表性 tiny LoRA 布局、TP 梯度与 checkpoint 恢复通过 |
| 真实模型 | 4B CPT 与 LoRA 的两种后端、7B MHA 和 8B Thinking 的 LoRA reference 后端通过短训练 |
| 性能 | 完成 RMSNorm 基准和真实 4B LoRA 短测，记录逐步耗时及离群值 |
| 集群规划 | 64 卡配置已准备；多机通信、吞吐和故障恢复待集群实测 |
| 扩展评测 | 7B/8B 全参数 CPT 的四卡显存、长上下文、长期收敛和领域 CPT 效果待验证 |

该次基线测试启用了真实 Megatron 参数解析集成，运行前后 50 个源码、测试、脚本、配置与构建文件保持一致。结果见 [独立测试报告](../artifacts/independent-final-suite.json) 和 [完整日志](../artifacts/independent-final-suite.log)。

环境和资产细节分别见 [运行环境](environment-audit.md)、[上游来源与数据](upstream-guide.md)。64 卡预设可以在本机生成 dry-run；训练入口会先检查可见设备数，并在设备不足时结束启动。

## 真实模型训练

六组功能验证统一采用对应模型的 HF 权重与 tokenizer 数据，BF16、序列长度 64、micro batch 1、global batch 4、full recompute。每组均核验四个 rank 完成、checkpoint 分片和恢复契约。

| 模型 / 任务 | TP / PP / DP | 后端 | 步数 | 各 rank 最大 allocated 的最大值（GiB） |
|---|---|---|---:|---:|
| 4B / CPT | 1 / 4 / 1 | reference | 2 | 20.646 |
| 4B / CPT | 1 / 4 / 1 | triton | 2 | 20.646 |
| 4B / LoRA | 2 / 2 / 1 | reference | 5 | 2.166 |
| 4B / LoRA | 2 / 2 / 1 | triton | 5 | 2.165 |
| 7B / LoRA | 4 / 1 / 1 | reference | 2 | 3.714 |
| 8B Thinking / LoRA | 4 / 1 / 1 | reference | 2 | 3.970 |

allocated 是 PyTorch 跟踪的分配峰值，驱动还会占用额外内存。4B CPT 的最大值出现在 rank 3，约 20.65 GiB；两个后端均保存 4 个模型分片和 4 个优化器分片，迭代记录为 2。这两步用于检查训练和保存流程。

LoRA 结束检查确认所有 rank 的基座参数保持冻结，全部 B 参数有限且非零：4B 每 rank 124/124、7B 为 112/112、8B Thinking 为 128/128。8B Thinking TP4 同时覆盖 TP 大于 KV 组数 2 的 QKV 激活 all-gather 路径。

[训练结果汇总](../artifacts/training-validation.json) 包含这六组功能验证及两组反序性能复测，共八个有效真实模型 run，提供逐步 loss、梯度、耗时、内存和 checkpoint 文件记录。

### LoRA 重计算检查

冻结 embedding 后，MCore 的 full recompute 需要可微的 activation 输入，才能将梯度传入 checkpoint 内的 adapter。训练入口为这段 activation 开启梯度，基座继续冻结，前向值保持一致。

四卡 MCore schedule 回归覆盖 TP4/PP1 和 TP2/PP2：前者每 rank 的 16/16 个 B、后者每 rank 的 8/8 个 B 均有有限非零梯度并更新，包含首个 PP stage；基座参数逐值不变、梯度为空，eval/no_grad 行为正常。回归日志见 [TP4](../artifacts/lora-recompute-tp4.log) 和 [TP2/PP2](../artifacts/lora-recompute-tp2pp2.log)。

每次 fresh LoRA 训练结束后，程序检查全部 B 和基座冻结状态。这项检查利用 B 的零初始化；本次恢复验收采用连续训练对照。此前有三组真实 LoRA 运行缺少这段梯度入口，导致全部或首阶段 adapter 保持初值，已在 [无效运行记录](../artifacts/invalid-lora-recompute.json) 中标记。

## 模型前向对齐

单卡、8 tokens、BF16，以 HF 实际 eager attention 类为参照。通过条件为 logits 相对 L2 < 0.03、loss 绝对差 < 0.1、top-1 一致率 ≥ 0.875。

| 模型 / 后端 | logits 相对 L2 | top-1 一致率 | 报告 |
|---|---:|---:|---|
| 4B reference | 0 | 1 | [结果](../artifacts/parity-4b-reference.json) |
| 4B Triton | 0.0119137 | 1 | [结果](../artifacts/parity-4b-triton.json) |
| 7B reference | 0 | 1 | [结果](../artifacts/parity-7b-reference.json) |
| 8B Thinking reference | 0.00496535 | 1 | [结果](../artifacts/parity-8b-reference.json) |

四组均通过，4B/7B reference 的逐层输出一致。报告记录实际 HF attention 类和源码版本。后续 DW 舍入与 LoRA activation 梯度修正保留了 CPT/eval 的前向计算，相关回归记录单独保存。早期使用不同 attention 后端的对照见 [Flash 对照记录](../artifacts/parity-4b-vs-hf-flash-unmatched.json)。

## 并行与恢复

| 检查 | 已覆盖内容 |
|---|---|
| tiny CPT / reference | DP4、DP2TP2、DP2PP2、TP4、TP2PP2、PP4，各运行 2 步 Adam 并保存 checkpoint |
| tiny LoRA | reference 的 DP2TP2、TP2PP2、PP4（tied embedding），以及 Triton 的 TP2PP2 短训练 |
| TP Column/Row LoRA | 与 dense 对照的前向、A/B 与输入梯度、基座冻结、adapter 保存恢复、非零 dropout 随机流 |
| distributed optimizer | tiny 短训练，以及真实 4B CPT 的完整保存 |
| Megatron 参数解析 | 4B PP4 的 16/15/15/16 分层与 tied embedding |
| 完整恢复 | TP2PP2 的 2→3 步恢复，对照连续 3 步训练 |

恢复对照中，四个分片的模型、Adam 和 master tensors 逐元素相同，adapter 继续更新、基座保持冻结。详情见 [checkpoint 恢复报告](../artifacts/checkpoint-resume-verification.json)。初轮 tiny 拓扑测试记录的是编排机制，最终模型实现的验证采用上文真实权重训练。

恢复入口检查 tracker、模型与优化器分片，以及 checkpoint 根目录的语义契约。契约记录模型、LoRA 超参、精度、并行和调度设置；迭代数、输出路径以及 reference/triton 后端可调整。不同后端的长程数值轨迹另行评估，操作步骤见 [配置与运行](configuration.md)。

## RMSNorm 数值与性能

数值测试覆盖 30 组实际形状、精度和 weight 组合，按原有容差通过。CoreX 归约需要先将中间乘积存为 weight dtype，再做 FP32 归约，以保留低精度舍入。DW 路径因此增加一次 kernel launch，以及 `rows × hidden × sizeof(weight dtype)` 的临时空间，在需要 weight gradient 时分配。

三种 BF16 hidden size 的 output 和 DW 误差均为 0，DX 在容差内。测试结果见 [DW 测试日志](../artifacts/ops/final-dw-tests.log) 和 [逐形状诊断](../artifacts/ops/dw-diagnostic-materialized.jsonl)。

性能测量独占一张卡，使用 rows=64、BF16、10 次预热、30 个样本，包含 DW 物化成本。设备事件反映 kernel 执行时间，同步墙钟反映调用到同步完成的耗时。

| hidden | 前后向设备中位数：reference / Triton（ms） | 同步墙钟中位数：reference / Triton（ms） | 报告 |
|---:|---:|---:|---|
| 2560 | 0.20093 / 0.03121 | 0.66592 / 0.52716 | [样本](../artifacts/ops/benchmark-final-clean-hf64-2560.json) |
| 3584 | 0.23532 / 0.03455 | 0.66256 / 0.49958 | [样本](../artifacts/ops/benchmark-final-clean-hf64-3584.json) |
| 4096 | 0.17788 / 0.02267 | 0.61123 / 0.47141 | [样本](../artifacts/ops/benchmark-final-clean-hf64-4096.json) |

这组单算子前后向测量中，同步墙钟下降约 21%–25%。完整训练的变化见下一节。

## 4B LoRA 端到端性能

性能短测使用 TP2/PP2/DP1、序列长度 64、GBS 4、seed 1234、dropout 0。每轮重新加载 HF 权重并将 B 初始化为零，全部 rank 的更新检查通过。两后端采用相同的模型、训练代码和数据，checkpoint 在末尾保存；iteration time 单独记录训练步耗时。

| 执行顺序与采样 | reference / Triton 中位数（ms） | reference / Triton 均值（ms） |
|---|---:|---:|
| reference→Triton，各 5 步，取后 3 步 | 1758.60 / 1704.70 | 1764.97 / 1868.20 |
| Triton→reference，各 12 步，取后 10 步 | 1777.95 / 1712.60 | 1801.81 / 1762.61 |

反序复测的步时中位数下降 3.68%，均值下降 2.18%；p90（线性插值）为 reference 1813.97 ms、Triton 1778.65 ms。两轮都有离群值：首轮 Triton 均值高于 reference，反序复测中也分别出现 2218.3 ms 和 2025.2 ms 的 Triton、reference 长步。

这组配置的短测显示中位步时改善，长期稳定性仍需更长训练评估。原始采样、统计定义和运行条件见 [训练结果汇总](../artifacts/training-validation.json) 中的 `reverse_order_lora_4b_experiment`。

## 共享源码与代码风格

各工作区共享读取 `/private/Megatron-LM`，`zbc` 工作区的旧位置保留兼容软链接。工程的 26 个 Python 文件按 79 列 PEP 8 规则整理，工具配置见 [pyproject.toml](../pyproject.toml) 和 [setup.cfg](../setup.cfg)。Black、pycodestyle、编译与 AST 审阅通过；逻辑、运行时字符串及导入时序保持一致，docstring 仅调整换行和空白。详情见 [静态检查报告](../artifacts/independent-pep8-static.json)。

本轮独立回归执行 31 项测试，全部通过，无跳过，unittest 耗时 141.030 秒。检查前后 55 个源码、配置和工具文件保持一致，结果见 [回归报告](../artifacts/independent-pep8-suite.json) 和 [完整日志](../artifacts/independent-pep8-suite.log)。
