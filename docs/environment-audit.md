# 天数运行环境

这套环境在 2026-10-29 完成四卡验证。训练工作区为 `/private/zbc`，共享源码位于 `/private/Megatron-LM`，共享权重位于 `/private/models`。

## 硬件与软件

服务器运行 Ubuntu 24.04.2、x86_64、Python 3.12.3，配有四张 Iluvatar BI-V150，每卡 32 GiB。CoreX 和驱动版本为 4.4.0，`/usr/local/corex` 指向 `/usr/local/corex-4.4.0`。

| 组件 | 实测版本 |
|---|---|
| torch | 2.7.1+corex.4.4.0 |
| triton | 3.1.0+corex.4.4.0 |
| torchvision / torchaudio | 0.22.1+corex.4.4.0 / 2.7.1+corex.4.4.0 |
| transformers / peft / accelerate | 4.57.6 / 0.18.1 / 1.11.0 |
| flash-attn / apex | 2.6.3+corex.4.4.0 / 0.1+corex.4.4.0 |
| bitsandbytes | 0.43.3+corex.4.4.0 |
| Megatron-LM | [工程固定版本](https://github.com/NVIDIA/Megatron-LM/commit/5be9626709af2722333bf54797c954c09edeada3) |

训练直接使用 Megatron 源码和 CoreX 运行库。Transformer Engine、DeepSpeed 未安装。厂商接口报告的 CUDA 10.2 来自兼容层；复现环境应使用相同的厂商镜像、驱动及 CoreX 包。完整版本记录见 [环境清单](../environment.lock.json)。

## 初始化与检查

在工程根目录执行：

```bash
cd /private/zbc/jiuge-training
source scripts/env.sh
bash scripts/bootstrap.sh
python3 scripts/doctor.py
torchrun --standalone --nproc_per_node=4 scripts/collective_smoke.py
```

[环境脚本](../scripts/env.sh) 将 CoreX 库目录加入 `LD_LIBRARY_PATH`，保留已有设置，可重复加载。普通 SSH shell 缺少这项设置时，`import torch` 会报 libcudart 或 libcublas 加载错误。

[构建脚本](../scripts/bootstrap.sh) 将 pybind11 2.13.6 安装到工程私有 `.deps`，供 Megatron CPU dataset helpers 编译使用。版本与下载校验信息集中在 [构建依赖清单](../requirements-bootstrap.txt)，厂商 Python 包沿用平台安装。

环境检查已确认四卡枚举、torch 设备访问，以及 `megatron.core`、`GPTModel`、`megatron.training` 导入正常。源码仓库完整性与固定版本检查通过。四 rank 通信测试覆盖 `all_reduce`、`all_gather` 和 rank 0→1 的 `send/recv`；脚本按 `LOCAL_RANK` 绑定设备，设置 30 秒进程组超时，并在结束时销毁进程组。

训练、数值和恢复结果见 [验证记录](validation.md)。

## 扩展到集群

目标布局为 DP2 × PP4 × TP8，共 64 卡。按每节点 8 卡部署，需要 8 个节点：TP8 位于节点内，每个 DP 副本跨 4 个 PP 节点。四卡环境已验证本地进程组与模型切分；集群验证还需覆盖网卡选择、IXCCL/NCCL 参数、rank placement、跨机带宽和故障恢复。

厂商运行库参考 [天数 ix-GPU-Operator 指南](https://developer.iluvatar.com/docs/generaldocs_ix_gpu_operator)，通信接口参考 [PyTorch distributed](https://docs.pytorch.org/docs/stable/distributed.html)。
