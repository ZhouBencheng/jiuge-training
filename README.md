# 天数九格训练实验台

在四卡 BI-V150 上运行九格 CPT、LoRA 指令微调和 RMSNorm 算子实验。训练使用固定版本的 Megatron，模型选项包括 FM9G-4B、FM9G-7B MHA、CPM-9G-8B MHA 和 8B 深思考版。

## 开始使用

服务器使用 CoreX 4.4、PyTorch 2.7.1+corex 和 Triton 3.1.0+corex，详见[环境清单](environment.lock.json)。在工程根目录执行：

```bash
cd /private/zbc/jiuge-training
source scripts/env.sh
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_CACHE_DIR="$PWD/.cache/triton"
bash scripts/bootstrap.sh
python3 scripts/doctor.py
```

`bootstrap.sh` 将 CPU 构建依赖装入工程 `.deps`，沿用平台提供的 torch/CoreX。首次检查四卡通信，再用 tiny 模型跑两步训练：

```bash
torchrun --standalone --nproc_per_node=4 scripts/collective_smoke.py
python3 -m jiuge_training.cli dry-run \
  --model tiny --task cpt-smoke --parallel dp4 --backend reference
python3 -m jiuge_training.cli train \
  --model tiny --task cpt-smoke --parallel dp4 --backend reference \
  --run-dir runs/cpt-smoke
```

Tiny 使用四层随机模型和合成 token，适合检查环境、通信和保存流程。真实权重训练从[配置与运行](docs/configuration.md)中的数据准备步骤开始。

## 任务与配置

| 任务 | 可训练参数 | loss 范围 |
|---|---|---|
| CPT | 全部基座参数 | 全部非 padding target |
| LoRA | 并行线性层的 adapter | 回答及 EOS，屏蔽指令 prompt |

配置分为 model、task、parallel、backend 四份 TOML。`validate` 展示组合后的配置，`dry-run` 打印启动命令，`train` 保存配置并启动 Megatron。

CPT 和 LoRA 共用以下六种四卡并行预设，通过 `--parallel` 选择：

| 预设 | DP | TP | PP |
|---|---:|---:|---:|
| `dp4` | 4 | 1 | 1 |
| `dp2-tp2` | 2 | 2 | 1 |
| `dp2-pp2` | 2 | 1 | 2 |
| `tp4` | 1 | 4 | 1 |
| `tp2-pp2` | 1 | 2 | 2 |
| `pp4` | 1 | 1 | 4 |

序列长度和 batch 大小需结合模型与每卡 32 GiB 显存调整。4B 的 62 层在 PP4 下分为 16/15/15/16 层。

Backend 可选 `reference` 或 `triton`，控制 RMSNorm 实现。其他 CUDA 融合默认关闭，参考 attention 使用显式 S² 矩阵。算子基准可单独运行：

```bash
PYTHONPATH=src python3 scripts/benchmark_ops.py \
  --rows 256 --hidden 2560 --dtype bfloat16
```

复现语料是官方中文版 Alpaca，共 48,818 条指令记录。CPT 与 LoRA 使用各自的数据格式和 loss mask；每个模型使用自己的 tokenizer。数据来源、固定版本和许可见[上游说明](docs/upstream-guide.md)。领域训练质量需另行评估。

## 工作空间

```text
/private/
├── Megatron-LM/             # 共享的固定版本上游源码
├── models/                  # 共享权重、原始归档及校验清单
└── zbc/                     # 训练工作区
    ├── Megatron-LM -> /private/Megatron-LM  # 旧路径兼容软链接
    ├── models -> /private/models          # 旧路径兼容软链接
    └── jiuge-training/
        ├── configs/         # model / task / parallel / backend
        ├── src/
        │   ├── jiuge_training/  # 数据、训练入口、模型、权重、LoRA
        │   └── jiuge_ops/       # Reference / Triton RMSNorm
        ├── scripts/         # 环境、通信与算子检查
        ├── docs/            # 操作说明与验证记录
        ├── data/            # 原始数据与处理结果
        ├── runs/            # 运行配置、命令与 checkpoint
        ├── logs/            # 原始日志
        ├── artifacts/       # 验证报告
        ├── .deps/           # 工程私有依赖
        └── .cache/          # 编译缓存
```

| 模型 | `--model` | 权重目录 | 层数 | Q / KV heads |
|---|---|---|---:|---:|
| 4B | `4b` | `/private/models/FM9G-4B/9G4B` | 62 | 40 / 4 |
| 7B MHA | `7b` | `/private/models/FM9G-7B/9G7B_MHA` | 28 | 28 / 28 |
| 8B MHA | `8b-mha` | `/private/models/CPM-9G-8B-MHA/sft_8b_v2` | 32 | 32 / 32 |
| 8B 深思考版（GQA） | `8b` | `/private/models/CPM-9G-8B/9g_8b_thinking` | 32 | 32 / 2 |

模型变体与发布来源见[上游说明](docs/upstream-guide.md#模型变体)。

权重在服务器下载并保留归档，下载校验和解压检查见[权重清单](/private/models/manifest.json)。Megatron 共享源码位于 `/private/Megatron-LM`，供各工作区读取；版本由[训练配置代码](src/jiuge_training/config.py)固定，启动时核验。模型和算子适配保存在训练工程中。

## 文档导航

| 文档 | 内容 |
|---|---|
| [配置与运行](docs/configuration.md) | 配置组合、数据准备、CPT/LoRA 命令、checkpoint 恢复 |
| [环境记录](docs/environment-audit.md) | 厂商软件栈、设备与通信检查 |
| [上游说明](docs/upstream-guide.md) | 模型、语料来源与许可 |
| [并行与 LoRA](docs/parallelism-and-lora.md) | 模型形状、布局约束和多机规划 |
| [模型适配](docs/model-adapter.md) | HF 权重映射、模型语义与 Python adapter 接口 |
| [验证记录](docs/validation.md) | 实测范围、性能数据和证据入口 |
