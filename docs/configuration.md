# 配置与运行

以下命令在服务器 `/private/zbc/jiuge-training` 根目录执行。先载入环境并准备私有构建依赖：

```bash
cd /private/zbc/jiuge-training
source scripts/env.sh
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
bash scripts/bootstrap.sh
```

固定版本的 Megatron 会编译 C++ 索引 helper。`bootstrap.sh` 将 pybind11 2.13.6 装入 `.deps`，wheel 保存在 `.cache/bootstrap`，按[依赖清单](../requirements-bootstrap.txt)校验。重复执行会检查既有版本，系统 torch/CoreX 保持原样。训练 CLI 自动补齐私有依赖与 Megatron 的搜索路径，共享源码默认位于 `/private/Megatron-LM`；手工执行 `dry-run` 输出的命令时，也需设置这两项 PYTHONPATH。

torchrun 各 worker 在 `TRITON_CACHE_DIR` 和 `TORCHINDUCTOR_CACHE_DIR` 基础路径下使用 `rank-N` 子目录，N 为全局 `RANK`；未指定基础路径时，分别使用工程内的 `.cache/triton/rank-N` 和 `.cache/torchinductor/rank-N`。

## 选择配置

一次运行组合四份 TOML：

| 配置 | 选项与用途 |
|---|---|
| `configs/model` | `4b`、`7b`、`8b` 为 HF 发布物，`8b-mha` 为旧版 MHA 归档；`tiny` 用于随机模型测试 |
| `configs/task` | `cpt`、`lora`，以及对应的 `*-smoke` 流程测试 |
| `configs/parallel` | DP、TP、PP、节点数和每节点进程数 |
| `configs/backend` | `reference` 或 `triton` RMSNorm |

```bash
python3 -m jiuge_training.cli validate \
  --model 4b --task lora --parallel tp2-pp2 --backend reference
python3 -m jiuge_training.cli dry-run \
  --model 4b --task lora --parallel tp2-pp2 --backend reference
```

`validate` 打印有效配置；`dry-run` 打印 torchrun 命令，保持文件和进程状态不变。`train` 核验固定 Megatron 版本、本节点可见卡数、数据与 checkpoint 条件，再保存配置并启动训练。版本定义见[配置代码](../src/jiuge_training/config.py)。

用 `--set SECTION.KEY=JSON` 覆盖字段，例如 `--set task.global_batch_size=4` 或 `--set 'task.data_path="data/processed/4b-lora-seq64"'`。字段名须已定义，梯度累积次数可显式指定，也可按下面的 batch 等式推导。

默认 run 目录是 `runs/<model>-<task>-<parallel>-<backend>`，checkpoint 放在其中的 `checkpoints`。`--run-dir` 修改运行目录，`task.save` 可单独指定 checkpoint 目录。每个新实验使用独立目录；恢复时显式设置 `task.load`。

### Batch 与并行

`GBS = MBS × DP × gradient_accumulation_steps`。默认 MBS=1、GBS=16，因此 DP4/DP2/DP1 分别累积 4/8/16 个 microbatch；累积次数至少为 PP。

四卡预设为 `dp4`、`dp2-tp2`、`dp2-pp2`、`tp4`、`tp2-pp2`、`pp4`。4B 的 62 层在 PP4 下分为 16/15/15/16 层，其他预设均匀分层。Q heads 和 FFN 维度须能被 TP 整除，TP 与 KV groups 须互为整数倍。TP 大于 KV groups 时，上游通过 QKV 激活 all-gather 完成计算，KV 可训练权重仍按 TP 分片。

64 卡预设为 8 节点、每节点 8 卡：4B/8B 使用 `future64-4b8b`（DP2×TP8×PP4），7B 使用 `future64-7b`（DP4×TP4×PP4）。7B 的 28 个 Q heads 不支持 TP8。多机启动需在各节点分别执行命令，使用不同 `--node-rank`、相同 `--master-addr`，并让配置、数据和 checkpoint 路径共享可见。

1/2 卡对照和 64 卡配置尚未实跑。`validate`、`dry-run` 可用于检查规划；`train` 会在 rendezvous 前检查本节点设备数量。布局与显存的实测范围见[验证记录](validation.md)，详细约束见[并行说明](parallelism-and-lora.md)。

### 模型与后端

模型从 `/private/models` 下各自的目录加载，路径表见[README](../README.md)。4B 共享输入 embedding 和输出权重；7B、8B 与默认 tiny 使用独立权重。`model.tie_word_embeddings` 和模型形状会与基座配置核对。

`reference` 使用 PyTorch RMSNorm，`triton` 使用工程内的 Triton 实现。其余 CUDA 融合默认关闭，后端加载失败会报错。真实任务默认使用 512 tokens、10 iterations 和 full recompute；CPT 启用 Megatron distributed optimizer。下文采用已实测的 64-token 短配置，方便检查训练与保存。

## 准备数据

复现语料为官方中文版 Alpaca，原始 48,818 条记录，固定来源和许可见[上游说明](upstream-guide.md)。每个模型、任务、序列长度分别准备数据，并使用该模型目录中的 tokenizer。

下面为 4B 准备 64-token CPT 和 LoRA 数据：

```bash
python3 -m jiuge_training.cli prepare-data \
  --source data/raw/alpaca_gpt4_data_zh.json \
  --output data/processed/4b-cpt-seq64 \
  --model-path /private/models/FM9G-4B/9G4B \
  --task cpt --seq-length 64 --verify-alpaca

python3 -m jiuge_training.cli prepare-data \
  --source data/raw/alpaca_gpt4_data_zh.json \
  --output data/processed/4b-lora-seq64 \
  --model-path /private/models/FM9G-4B/9G4B \
  --task lora --seq-length 64 --verify-alpaca
```

默认任务数据路径是 `data/processed/<model>-<task>`；采用上面的独立目录时，在训练命令中覆盖 `task.data_path`。7B/8B 同样使用各自模型目录和输出目录，训练时选择相应 `--model`。

准备过程先去除完全相同的记录，按 seed=1234、98/2 的比例划分训练与验证集，再分别编码。Tokenizer 只读取本地资产，自定义 tokenizer 代码通过 `trust_remote_code` 加载。

CPT 使用以下格式拼接，每条末尾加 EOS；有 input 时才加入输入段：

```text
### 指令：
{instruction}
### 输入：
{input}
### 回答：
{output}{EOS}
```

拼接后按固定长度切块，相邻块保留一个 token 重叠用于 next-token 目标，padding 不计入 loss。LoRA 分别编码 prompt 和回答，只对回答及 EOS 计算 loss。超长样本先截掉 prompt 左侧；回答超过窗口时保留前段并以 EOS 结束。这是复现语料的格式，领域 CPT 的语料选择和效果评估需另行开展。

输出目录中的 `manifest.json` 记录划分、监督 token 数、tokenizer/config、Transformers 版本与数据文件指纹，训练启动时重新核验。JSONL Dataset 将样本读入内存，并按 Megatron 样本索引循环；更大语料需要按容量更换输入实现。

服务器保留旧模型位置到 `/private/models` 的软链接。路径校验会解析软链接，因此既有数据清单仍可使用；历史 run、日志和 manifest 保留生成时的内容。

## 运行 CPT 与 LoRA

初次加载来自 `model.path`。CPT 更新基座全部参数，LoRA 冻结基座并训练并行线性层 adapter，rank、alpha、dropout 在 task 配置中设置。

4B CPT 使用 PP4，训练两步后保存完整 checkpoint：

```bash
python3 -m jiuge_training.cli train \
  --model 4b --task cpt --parallel pp4 --backend reference \
  --run-dir runs/my-4b-cpt-reference \
  --set task.seq_length=64 --set task.global_batch_size=4 \
  --set task.train_iters=2 --set task.save_interval=2 \
  --set task.eval_iters=0 \
  --set 'task.data_path="data/processed/4b-cpt-seq64"'
```

4B LoRA 使用 TP2PP2，下面的命令运行十二步并在末尾保存：

```bash
python3 -m jiuge_training.cli train \
  --model 4b --task lora --parallel tp2-pp2 --backend reference \
  --run-dir runs/my-4b-lora-reference \
  --set task.seq_length=64 --set task.global_batch_size=4 \
  --set task.train_iters=12 --set task.save_interval=12 \
  --set task.eval_iters=0 \
  --set 'task.data_path="data/processed/4b-lora-seq64"'
```

比较后端时改为 `--backend triton`，另设 run 目录，其余参数和数据保持相同。7B/8B 的已验收短训练使用 `--parallel tp4 --backend reference`、64 tokens、GBS=4 和两步训练，配合各自的 seq64 LoRA 数据。

运行目录会保存 `config.json`、`command.txt`、`source_manifest.json` 和每个 rank 的 `metrics-rankN.json`。Source manifest 绑定代码、有效配置、上游版本、环境与数据清单。Metrics 包含显存和运行时间，其中 wall time 包含加载与保存；比较迭代性能时使用日志中的 iteration time，单独处理预热和 checkpoint 开销。实测数据见[训练报告](../artifacts/training-validation.json)。

新 LoRA 训练结束后，每个 rank 检查全部零初始化的 `lora_B` 已变为有限非零值，并检查基座参数仍为 `requires_grad=False`。结果保存在 `training-validity-rankN.json` 和 metrics，缺失更新会使任务报错。检查在迭代计时结束后执行；恢复训练的更新检查标记为 `not_verified_for_resume`，本次验收使用连续训练对照验证恢复正确性。

### Tiny 流程检查

Tiny 是四层随机模型，使用小词表与合成 token，搭配 `cpt-smoke` 或 `lora-smoke`。真实模型搭配 `cpt` 或 `lora` 及准备好的数据。

```bash
python3 -m jiuge_training.cli train \
  --model tiny --task cpt-smoke --parallel dp4 --backend reference \
  --run-dir runs/cpt-smoke
python3 -m jiuge_training.cli train \
  --model tiny --task lora-smoke --parallel tp2-pp2 --backend triton \
  --run-dir runs/lora-smoke
```

## 恢复训练

CLI 使用 Megatron `torch` 格式的完整 checkpoint，恢复模型、优化器、RNG 和迭代数。以下例子将两步 tiny CPT 延长到第三步：

```bash
python3 -m jiuge_training.cli train \
  --model tiny --task cpt-smoke --parallel dp4 --backend reference \
  --run-dir runs/cpt-smoke-resume \
  --set 'task.load="runs/cpt-smoke/checkpoints"' \
  --set task.train_iters=3
```

真实模型恢复使用同样的 `task.load`，并带上原运行的模型、任务、并行和数据参数。恢复保留 checkpoint 调度状态，入口传入 `--use-checkpoint-opt-param-scheduler`。Constant LR/WD 可延长训练步数；更改学习率或 weight decay 时另起实验。

Checkpoint 根目录的 `jiuge_contract.json` 记录模型结构、共享权重、基座配置指纹、任务、LoRA rank/alpha/dropout、BF16、distributed optimizer、DP/TP/PP 和 LR/WD。入口检查契约、迭代记录及全部分片；缺失或不匹配会报错。契约以原子方式写入，已有内容须匹配，自定义 `task.save` 也适用。早期没有契约的实验 checkpoint 保留作历史证据。

恢复可修改迭代数、输出位置和日志设置，也可切换 reference/triton。两种实现的舍入差异可能影响长期轨迹。TP/PP/DP 保持 checkpoint 布局。

CLI 的保存与恢复对象是完整训练状态。单独导出和加载 adapter 使用 Python 的 `save_adapter` / `load_adapter`，调用方式见[模型适配](model-adapter.md)。

## 检查命令

配置、数据与 CLI 检查：

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_config*.py'
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_data*.py'
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_cli*.py'
```

在厂商环境中，还可执行固定上游的 CPU 参数解析检查：

```bash
PYTHONPATH=src JIUGE_MEGATRON_INTEGRATION=1 python3 -m unittest discover -s tests -p 'test_config_megatron.py'
```

该项核对 4B PP4 的 16/15/15/16 层及共享 embedding，使用上游参数 `--decoder-first-pipeline-num-layers`、`--decoder-last-pipeline-num-layers`。未设置环境变量时默认跳过。

已完成的模型、布局、后端和恢复测试统一记录在[验证文档](validation.md)，包括有效实验与保留用于诊断的早期 run。

### 旧版 8B MHA 的数据路径

`--model 8b-mha` 读取 `/private/models/CPM-9G-8B-MHA/sft_8b_v2`。这个目录保存原始 `.pt`、配置与词表，`prepare-data` 自动选择旧版 CPM9G tokenizer。CPT 和 LoRA 默认分别读取 `data/processed/8b-mha-cpt`、`data/processed/8b-mha-lora`，需要用该目录的词表重新准备数据。模型结构与加载方式见 [旧版 8B MHA 实验入口](model-adapter.md#旧版-8b-mha-实验入口)。
