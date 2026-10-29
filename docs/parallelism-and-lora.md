# 四卡布局与 64 卡规划

总卡数满足 `world_size = DP × TP × PP`。DP 复制模型处理不同数据，TP 在层内切分张量，PP 将模型层分配到不同 stage。选择布局时，先用 TP/PP 控制每卡显存，再用 DP 增加数据并行度。

## 四卡怎么选

| 预设 | DP | TP | PP | 主要用途 | 已有运行证据 |
|---|---:|---:|---:|---|---|
| `dp4` | 4 | 1 | 1 | 每卡完整模型，检查数据并行 | tiny CPT |
| `dp2-tp2` | 2 | 2 | 1 | 两个副本，各用两卡做层内切分 | tiny CPT、LoRA |
| `tp4` | 1 | 4 | 1 | 用四卡分担每层权重 | tiny CPT；真实 7B MHA、8B Thinking LoRA |
| `dp2-pp2` | 2 | 1 | 2 | 两条各含两个 stage 的流水线 | tiny CPT |
| `tp2-pp2` | 1 | 2 | 2 | 同时切分层内权重和模型层 | tiny CPT、LoRA；真实 4B LoRA |
| `pp4` | 1 | 1 | 4 | 按层分到四卡，减少层内通信 | tiny CPT、LoRA；真实 4B CPT |

真实模型短训练采用 BF16、序列长度 64、micro batch 1、global batch 4 和 full recompute。扩大序列或批量后，需要重新测量峰值显存。运行条件和结果见 [验证记录](validation.md#真实模型训练)，启动方法见 [配置与运行](configuration.md)。

单卡和双卡适合拆开检查模型与通信：

| 卡数 | DP × TP × PP | 检查重点 |
|---:|---|---|
| 1 | 1 × 1 × 1 | HF 权重导入、逐层输出和 logits |
| 2 | 2 × 1 × 1 | DP 梯度同步 |
| 2 | 1 × 2 × 1 | TP 线性层和 attention |
| 2 | 1 × 1 × 2 | PP 前后向与 stage 参数分配 |

这些布局均满足四种模型的静态切分条件，实际容量取决于任务、精度、优化器与激活占用。

## 模型形状决定切分方式

| 模型 | hidden | 层数 | Q heads | KV groups | FFN |
|---|---:|---:|---:|---:|---:|
| FM9G-4B | 2560 | 62 | 40 | 4 | 6400 |
| FM9G-7B MHA | 3584 | 28 | 28 | 28 | 18944 |
| CPM-9G-8B MHA | 4096 | 32 | 32 | 32 | 14336 |
| CPM-9G-8B Thinking | 4096 | 32 | 32 | 2 | 16384 |

TP 要求 Q heads、hidden 和 FFN 宽度能被 TP 整除。KV groups 与 TP 满足一方整除另一方的关系；当 TP 大于 KV groups 时，Megatron 通过 QKV 激活 all-gather 供各 rank 选择对应的 Q/K/V。

8B Thinking 的 TP4/KV2 就使用这条路径。QKV 可训练权重仍沿打包后的输出维度分片，反向通过 all-gather 汇总梯度。真实 TP4 LoRA 已检查全部 adapter B 更新；TP8 留在集群规划中。

PP 按层切分。4B 的 62 层在 PP4 下使用 **16 / 15 / 15 / 16**，预设设置首尾 stage 各 16 层，中间均分。7B 与 8B 的 PP4 分别为每 stage 7 层、8 层。不均匀首尾分层与 embedding/loss 独立计入 pipeline split 的方式互斥。

模型缩放、RoPE、权重打包及 LoRA 通信约定见 [模型适配](model-adapter.md)。本工程的 LoRA 挂在 TP 线性层上；Megatron 的 `q_lora_rank` / `kv_lora_rank` 则属于 MLA 架构内部的低秩投影，两者各自配置。

## 显存与训练配置一起看

BF16 基座权重约占参数量 ×2 bytes。全参数 Adam 加上 FP32 梯度、主参数和双动量，模型状态可接近 18 bytes/parameter；激活、通信 buffer、临时张量和 PP 阶段不均衡还会增加峰值。优化器分片与重计算会改变各卡占用。

LoRA 将梯度和优化器状态集中在 adapter 上，仍需保存基座和反向所需激活。本工程的参考 attention 会物化 S² 分数，序列长度对显存的影响尤其需要实测。

四张 32 GiB 卡上已完成 4B CPT，以及 4B、7B MHA 和 8B Thinking 的 LoRA 短训练。7B/8B 全参数 CPT 的四卡容量待验证。比较布局时应固定序列长度、micro/global batch 和重计算设置，记录各 rank 峰值中的最大值。

## 64 卡规划

下面是满足模型切分条件的集群预设，多机通信、吞吐和故障恢复仍需在目标集群验证。

| 模型 | 预设 | DP × TP × PP | 切分说明 |
|---|---|---|---|
| 4B | `future64-4b8b` | 2 × 8 × 4 | Q heads 40；KV4 使用激活 all-gather；PP 为 16/15/15/16 |
| 7B | `future64-7b` | 4 × 4 × 4 | Q heads 28 可按 TP4 切分；每 stage 7 层 |
| 8B Thinking | `future64-4b8b` | 2 × 8 × 4 | Q heads 32；KV2 使用激活 all-gather；每 stage 8 层 |

7B 的 28 个 Q heads 无法按 TP8 等分，因此采用 TP4/PP4/DP4。

若每节点 8 卡，4B/8B 可以将 TP8 放在节点内，PP4 跨四个节点组成一个副本，两个 DP 副本共八个节点。7B 的 TP4 可在每节点安排两个 TP 组；具体 rank placement 结合调度器和节点互联确定。

## CoreX 上的算子选择

工程使用与 CoreX 配套的 PyTorch、通信库和扩展，环境验收见 [运行环境](environment-audit.md)。引入新 kernel 时，先与参考实现核对数值和梯度，再测训练中的热点与收益。

算子通过 `ModuleSpec` 接入，厂商实现与训练循环、优化器、DP/PP 编排分开。`backend.rmsnorm` 选择 reference 或 Triton RMSNorm；接口、DW 中间舍入成本和注意力的显存特性见 [算子替换接口](model-adapter.md#算子替换接口)。精确版本与运行配置保存在 [训练结果汇总](../artifacts/training-validation.json) 中。
