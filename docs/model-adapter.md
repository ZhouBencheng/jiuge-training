# 九格模型与 LoRA 适配

模型适配由 [model.py](../src/jiuge_training/model.py)、[weights.py](../src/jiuge_training/weights.py) 和 [lora.py](../src/jiuge_training/lora.py) 组成，分别负责层结构、权重导入和并行 adapter。旧版 CPM9G 的配置与词表接口位于 [legacy.py](../src/jiuge_training/legacy.py)。布局选择见 [四卡布局与 64 卡规划](parallelism-and-lora.md)。

## 模型结构

以下三个 HF 发布物都使用 pre-RMSNorm、SiLU 门控 MLP 和非交错 RoPE。各自的缩放和注意力约定如下。

| 模型 | 层数 | Q / KV heads | head 维度 | 模型约定 |
|---|---:|---:|---:|---|
| FM9G-4B | 62 | 40 / 4 | 64 | embedding ×12；两个残差分支各 ×1.4/√62；输出隐藏状态 ÷10；embedding 与输出头共享权重 |
| FM9G-7B MHA | 28 | 28 / 28 | 128 | Q/K/V 含 bias；普通残差；embedding 与输出头独立 |
| CPM-9G-8B thinking | 32 | 32 / 2 | 128 | embedding ×12；残差分支各 ×1.4/√32；输出隐藏状态 ÷16；逐维 LongRoPE |

4B 的配置类将 `tie_word_embeddings` 默认为 true，checkpoint 省略 `lm_head.weight`，加载时复用 embedding 权重。4B/8B 的 Q/K 旋转在 FP32 中完成，再转回激活 dtype；7B 沿用其低精度旋转顺序。

8B LongRoPE 支持已核对的 amplitude=1、short_factor 与 long_factor 相同的配置。模型接口使用 CP=1、关闭 sequence parallel 的普通序列；其他 LongRoPE 变体、packed sequences、MLP bias 和 sliding window 会在配置检查时报错。

## 加载、缩放与前向核对

`model.configure(config)` 接收 CLI 合并配置；`model_provider(pre_process, post_process, config=None, pg_collection=None)` 创建当前 TP/PP 分片。HF 发布物首次运行从原 HF 权重加载基座，恢复训练从 Megatron checkpoint 加载。`model.smoke=true` 创建随机初始化的 tiny 模型，用于验证训练编排。

Safetensors 通过 mmap slice 读取，7B 的单个 `.bin` 使用 `torch.load(weights_only=True, mmap=True)`。每个 rank 按以下规则读取自己的参数：

- QKV 按 `[一组 Q heads, K head, V head]` 打包，再按 TP 输出维度切分。
- 门控 MLP 保存本 rank 的 gate/up 分片，输出投影按输入列切分。
- PP 层通过 `layer.layer_number` 对应全局 HF 层号。
- 词表补齐行置零，并在 logits 中屏蔽。

加载前核对配置中的层数、hidden、Q/KV heads、FFN、词表和权重共享设置；加载时检查 HF 键集合、目标参数形状及本 PP stage 的参数覆盖。

当 TP 大于 KV 组数时，Megatron 将分片 QKV 激活 all-gather，再选择各 rank 的 Q heads 和对应 KV 组。可训练权重仍沿打包后的输出维度分片，all-gather 的反向负责汇总梯度。8B 的 TP4/KV2 已覆盖真实 LoRA 短训练。

前向核对采用单卡、8 tokens、BF16，并显式选择 HF eager attention。4B/8B 配置类会自动选择 Flash 后端，因此测试会在配置构造后设置 eager，并核验实际 attention 类。预设通过条件为 logits 相对 L2 < 0.03、loss 绝对差 < 0.1、top-1 一致率 ≥ 0.875。

| 模型 / RMSNorm | logits 相对 L2 | HF / Megatron loss | top-1 一致率 |
|---|---:|---|---:|
| 4B / reference | 0 | 13.788101 / 13.788101 | 1.0 |
| 4B / Triton | 0.011914 | 13.788101 / 13.791131 | 1.0 |
| 7B / reference | 0 | 12.402442 / 12.402442 | 1.0 |
| 8B / reference | 0.004965 | 12.614639 / 12.615602 | 1.0 |

4B/7B reference 在该输入下逐层输出及最终 logits 一致，另两项在预设容差内通过。完整报告和训练覆盖范围见 [验证记录](validation.md#模型前向对齐)；长上下文与长期收敛属于后续评测。

## LoRA 梯度与 checkpoint

Adapter 接入 attention QKV、attention 输出投影和 MLP 两个线性层。全部基座参数冻结，B 从零初始化，因此初始前向与基座一致。

| 并行线性层 | A / B 切分 | 梯度通信 |
|---|---|---|
| Column parallel | A 复制，B 按输出分片 | A 输出的 copy-backward 汇总 TP 梯度，使 A 和输入梯度包含各输出分片的贡献 |
| Row parallel | A 按输入分片，B 复制 | A 输出先做 TP reduce 再乘 B，使复制 B 的梯度一致 |

CUDA dropout 的 column 路径使用 Megatron data-parallel RNG stream，让复制 A 在 TP ranks 间使用一致掩码；row 路径使用 model-parallel stream，让不同输入特征分片使用独立掩码。

full recompute 使用重入 checkpoint，需要可微的 activation 输入。LoRA 训练时，首个 PP stage 在 embedding 缩放后为输出开启 `requires_grad=True`，为 checkpoint 内的 adapter 提供梯度入口。这个设置只作用于训练 activation，基座继续冻结，eval/no_grad 行为和前向值保持一致。

四卡 tiny 回归覆盖 TP4/PP1 与 TP2/PP2。两步后，每 rank 分别有 16/16、8/8 个 B 获得有限非零梯度并更新，包含首个 PP stage；基座逐值不变且梯度为空。真实训练也检查每个分片的 adapter 更新覆盖，结果见 [LoRA 重计算检查](validation.md#lora-重计算检查)。

| 入口 | 保存内容 | 恢复用途 |
|---|---|---|
| CLI `task.save` / `task.load` | 基座、adapter、优化器与 RNG 的完整 Megatron `torch` checkpoint | 继续训练 |
| Python `jiuge_training.lora.save_adapter` / `load_adapter` | Adapter 参数、配置和并行布局 | 在相同基座与 TP/PP 布局上加载 adapter |

Python 接口接收解包 DDP/Float16Module 后的本 stage GPTModel。每个 TP/PP rank 使用独立文件名，DP 保存一个副本即可；调用者提供同一基座权重。加载时核对配置、布局、键和形状。CLI 使用完整 checkpoint；adapter-only 导出通过上述 Python 接口完成。跨 TP/PP 布局的 adapter 重分片待验证。

## 算子替换接口

层结构通过 `ModuleSpec` 组合。替换算子时保留输入输出、TP 通信和 autograd 约定，厂商实现放在独立算子模块中。

`backend.rmsnorm` 选择 `jiuge_ops.ReferenceRMSNorm` 或 `TritonRMSNorm`。两者接受 `hidden_size`、`eps`、`config`，提供 `forward(x)`，参数名为 `weight`。RMSNorm 按模型约定先在 FP32 中计算方差和归一化，再舍入到输入 dtype 后乘 weight。库将最终输出保持为输入 dtype；混用 FP32 weight 时，这与 HF 的输出类型提升有所区别。训练采用统一 BF16 权重。

权重梯度 DW 的逐元素乘积先按 weight dtype 舍入，再做 FP32 归约。CoreX 会省略同一 kernel 内 multiply→extend→reduce 的中间舍入，因此实现用独立 kernel 将乘积写入 weight dtype 临时张量。这增加一次 kernel launch，以及 `rows × hidden × sizeof(weight dtype)` 的临时空间，仅在需要 weight gradient 时执行。算子基准已包含这项成本，数值和性能见 [RMSNorm 验证](validation.md#rmsnorm-数值与性能)。

`HFCompatibleAttention` 使用 PyTorch matmul 后单独缩放，保留 HF 的低精度计算顺序。它会物化 S² 注意力分数，长序列需要重新测量显存。`backend.rmsnorm` 只切换归一化算子，注意力实现保持一致。

## 验证命令

```bash
source scripts/env.sh
export PYTHONPATH="/private/Megatron-LM:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m unittest discover -s tests -p 'test_model_weights.py'
python -m unittest discover -s tests -p 'test_lora.py'
torchrun --standalone --nproc-per-node=4 tests/test_lora_parallel.py
torchrun --standalone --nproc-per-node=4 tests/test_lora_recompute.py --pp 1
torchrun --standalone --nproc-per-node=4 tests/test_lora_recompute.py --pp 2
CUDA_VISIBLE_DEVICES=0 USE_TF=0 python tests/test_model_parity.py \
  --model-path /private/models/FM9G-4B/9G4B \
  --output artifacts/parity-4b-reference.json
```

前向测试比较 logits、loss、top-1 和逐层输出，失败时返回非零；`--dtype float32` 可用于结构诊断。各次运行的配置、源码版本和原始结果保存在 [验证记录所链接的机器报告](validation.md)。

## 旧版 8B MHA 实验入口

`8b-mha` 对应官方 `sft_8b_v2` 归档，使用原 BMTrain `.pt`、`config.json` 和 `vocab.txt`。它与上文的 8B Thinking 分开选择：32 层、hidden 4096、Q/KV 各 32、FFN 14336、词表 119696；原配置 `scale=false`，embedding 与输出头独立。归档权重文件为 FP16，训练精度由 task 配置选择，默认采用 BF16。

入口通过 `checkpoint_format="cpm9g-bmtrain"` 选择原格式。加载器 mmap 读取单个 `.pt`，核对全部原始键，再将其映射到现有 TP/PP 权重加载逻辑；原文件保持原格式。配置字段只在内存中转换为层构造参数。

旧版 eager attention 在 RoPE 前将 Q、K 各除以 head_dim 的四次方根，随后直接计算注意力分数。RoPE base 为 10000，频率在 FP32 生成，cos/sin 转为激活 dtype 后旋转；softmax 保持原激活 dtype。它使用专用 attention 接口，并复用 RMSNorm、MLP、LoRA 和 checkpoint 编排。

词表采用最长前缀匹配和 UTF-8 byte fallback，EOS 为 2，padding 使用 EOS。数据准备读取归档原词表并记录指纹。单次编码上限为 20480 字符，超过时报错。

[实际权重检查](../artifacts/8b-mha-weights-verification.json) 已核对全部 291 个原始张量的键与形状。`tests/test_legacy_parity.py` 以官方纯 PyTorch native layers 的 causal eager 路径为参考，使用完整权重、单卡 8 tokens 分别检查 [FP16 原精度](../artifacts/parity-8b-mha-float16.json) 和 [BF16 训练精度](../artifacts/parity-8b-mha-bfloat16.json)；两组逐层输出和最终 logits 均一致。CPU fixture 另行覆盖配置、词表、权重映射、attention 与恢复契约。
