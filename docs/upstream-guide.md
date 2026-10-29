# 上游来源与数据

工程以 [固定版本的 Megatron-LM](https://github.com/NVIDIA/Megatron-LM/commit/5be9626709af2722333bf54797c954c09edeada3) 和服务器的 CoreX 环境为运行基础。九格官方资料用于核对模型结构、原训练流程和数据格式。

## 九格训练资料

[README_ALL](https://www.osredm.com/jiuyuan/CPM-9G-8B/tree/d0e90e99ff6e34c58a088b9cb052db92a6afffb3/quick_start_clean/readmes/README_ALL.md) 使用 BMTrain，以 `torchrun` 启动 `9G-Train/apps/cpm9g/pretrain_cpm9g.py`。对应版本的环境和 LoRA 指南一并保存在研究缓存 `.cache/upstream-osredm/repo`。

| 资料 | 使用环境 | 参考内容 |
|---|---|---|
| [原环境指南](https://www.osredm.com/jiuyuan/CPM-9G-8B/tree/d0e90e99ff6e34c58a088b9cb052db92a6afffb3/quick_start_clean/readmes/README_ENV.md) | Python 3.8.10、PyTorch 1.13.1、BMTrain 0.2.3.post2、flash-attn 2.0.8、CUDA 11.6 | 原训练流程与依赖关系 |
| [FM_9G 分支文档](https://www.osredm.com/jiuyuan/CPM-9G-8B/tree/FM_9G/FM_9G/apps/fm9g_70b/README.md) | Python 3.10.16、PyTorch 2.3.0、flash-attn 2.7.0.post2、Transformers 4.44.0 | 后续模型的 BMTrain 实现 |
| [本工程环境](environment-audit.md) | Python 3.12.3、PyTorch 2.7.1+corex.4.4.0、Triton 3.1.0+corex.4.4.0 | BI-V150 实测与复现 |

原指南将文本转为 `{"input":"","output":"文本"}`，指令样本的 prompt 和答案分别放入 `input`、`output`，再通过 `data_binarize.py` 生成 data/meta.bin。本工程在 JSONL 中直接保存 token 和 loss mask：CPT 对文本序列计算 loss，LoRA 指令微调只训练回答与 EOS。

[官方 LoRA 指南](https://www.osredm.com/jiuyuan/CPM-9G-8B/tree/d0e90e99ff6e34c58a088b9cb052db92a6afffb3/quick_start_clean/readmes/README_LORA.md) 使用 `--delta-tuning --delta-type lora`，示例 rank/alpha 为 64、dropout 为 0.05，目标层为 project_q/project_v/project_k/w_0/w_1/w_out，并提供 delta 合并脚本。本工程采用 Megatron TP/PP 线性层适配，接口和权重映射见 [模型适配](model-adapter.md)。

工程支持原生 PT 与 HF 两种发布格式。模型适配处理 QKV/MLP 布局、embedding 与残差缩放、RoPE 和 logits；8B 深思考版使用其发布版本的 LongRoPE 配置。

## 模型变体

| 选项 | 官方模型 | 注意力结构 | 发布来源 |
|---|---|---|---|
| `4b` | FM9G-4B | GQA，Q/KV heads 为 40/4 | [模型发布清单](https://www.osredm.com/jiuyuan/CPM-9G-8B/src/branch/FM_9G/README.md) |
| `7b` | FM9G-7B MHA | MHA，Q/KV heads 为 28/28 | [7B 推理说明](https://www.osredm.com/jiuyuan/CPM-9G-8B/src/branch/FM_9G/quick_start_4_7_70/inference_7b.md) |
| `8b-mha` | 早期 CPM-9G-8B v2 | MHA，Q/KV heads 为 32/32 | [原训练指南](https://www.osredm.com/jiuyuan/CPM-9G-8B/tree/d0e90e99ff6e34c58a088b9cb052db92a6afffb3/quick_start_clean/readmes/README_ALL.md)的 `8b_v2` 下载项 |
| `8b` | CPM-9G-8B v2 深思考版 | GQA，Q/KV heads 为 32/2 | [8B V2 推理说明](https://www.osredm.com/jiuyuan/CPM-9G-8B/src/branch/FM_9G/quick_start_4_7_70/inference_8b.md) |

早期 8B MHA 使用官方 [sft_8b_v2.zip](https://qy-obs-6d58.obs.cn-north-4.myhuaweicloud.com/sft_8b_v2.zip)，包含原生 PT 权重、配置与 vocab.txt。新版 8B 深思考版使用 HF 发布格式，两者分别通过 `8b-mha` 和 `8b` 选择，加载方式见[模型适配](model-adapter.md)。

本次查阅了官方 master、FM_9G、FM9G-V 分支及 v1.0.0 标签，尚未发现 7B Thinking 的独立公开权重下载项。[官方发布清单](https://www.osredm.com/jiuyuan/CPM-9G-8B/src/branch/FM_9G/README.md)列出的 7B 为 MHA，深思考版下载项为 8B。

## 中文指令数据

数据来自 [Alpaca GPT-4 中文数据的固定版本](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/38b93b75a80b019cc3aad362ca6f5bb0c3908959/data/alpaca_gpt4_data_zh.json)，共 48,818 条，文件约 35.1 MB。每条记录包含字符串字段 `instruction`、`input`、`output`。原文件位于 `data/raw/alpaca_gpt4_data_zh.json`，版本、大小和校验信息见 [数据来源记录](../data/raw/manifest.json)。

指令微调对回答与 EOS 计算 loss；CPT 将标题、指令、输入和回答拼接为文本，用于复现继续预训练流程。领域 CPT 的语料质量另行评估。每套处理结果的 manifest 记录划分、序列长度、处理格式、tokenizer 和数据文件校验信息，准备方法见 [配置与运行](configuration.md)。

上游 [README](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/README.md) 将代码许可 Apache-2.0 与数据许可 CC BY-NC 4.0 分别列出，并注明数据用于研究。实验保留这份来源与许可说明。

## 模型权重

四套权重均由服务器从官方资料引用的地址下载，保存在共享目录 `/private/models`。下载地址、目录与归档校验信息见 [模型清单](/private/models/manifest.json)，服务器保留原始归档。

| 模型 | 发布格式 | 权重结构 |
|---|---|---|
| FM9G-4B | HF safetensors | 2 个分片，560 个索引张量 |
| FM9G-7B MHA | HF PyTorch | 单个 `pytorch_model.bin` |
| CPM-9G-8B MHA | 原生 PT | 单个 `.pt`、`config.json` 与 `vocab.txt` |
| CPM-9G-8B v2 深思考版 | HF safetensors | 4 个分片，291 个索引张量 |

模型前向对齐、训练与恢复的实测结果见 [验证记录](validation.md)。
