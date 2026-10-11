# 44 - vllm_py310 环境安装记录（含失败路线，勿重试）

> 2026-10-08 整理自会话记录。远程机：共享 8×A800 服务器，驱动 **550.90.07
> （CUDA 12.4）**，**不能动驱动**。env 用途：vLLM serving + v8 插件 + triton
> microbench（P1.2 的 kernel 就在这个 env 里跑的）。

## 最终可用组合（唯一验证过的工作点）

| 组件 | 版本 | 来源 |
|---|---|---|
| Python | 3.10 | conda |
| torch | **2.10.0+cu128** | PyPI（vllm 0.19.1 的依赖 pin 自动拉对） |
| vllm | **0.19.1** | 清华镜像 |

关键机制：torch ≤2.10 对驱动做 **minor 版本兼容**（cu128 runtime 可跑在 CUDA
12.4 驱动上，gemma4_infer env 实测）；torch **2.11+ 启动时硬性检查驱动 ≥12.8**，
直接拒绝，无解。

## 安装命令（从头建）

```bash
conda create -n vllm_py310 python=3.10 -y
conda activate vllm_py310
pip install vllm==0.19.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

如果 env 里已有装错的 vllm/torch（见"失败路线"），先清：

```bash
pip uninstall -y vllm torch torchvision torchaudio
# 换 torch 时配套 nvidia-* 包必须整体重装（见坑 2），最干净的办法是删 env 重建：
# conda remove -n vllm_py310 --all -y，然后按上面从头建
```

## 验证（两步都必须过）

```bash
# 1. CUDA + 驱动兼容 + vllm import
python -c "import torch; torch.zeros(1).cuda(); import vllm; print('ok', vllm.__version__)"
# 期望: ok 0.19.1

# 2. Qwen3.6 arch 注册（Qwen3_5Moe 很新，老版本 vllm 没有）
python -c "from vllm.model_executor.models.registry import ModelRegistry; print('Qwen3_5MoeForConditionalGeneration' in ModelRegistry.get_supported_archs())"
# 期望: True
```

triton 随 vllm 依赖链安装，无需单独装（microbench 直接可用）。

## 失败路线（全部实测撞墙，勿再试）

| 路线 | 死因 |
|---|---|
| `pip install vllm --pre` nightly（0.30.x） | 拉 torch 2.11+cu13（libcudart.so.13），torch 2.11 硬性拒绝 <12.8 驱动 |
| torch 2.11/2.13 + cu128/cu126 手动装 | torch ≥2.11 启动时驱动检查是硬性的，CUDA runtime > 驱动即拒，无 minor 兼容 |
| 驱动 runfile `--extract-only` 拿用户态 libcuda 盖 LD_LIBRARY_PATH | 技术上可行（纯解包不动系统），但共享机器上不碰驱动话题，主动放弃 |
| vllm 0.9.2 + torch 2.6 方案 | 曾作为备选，后被 0.19.1 路线取代（0.9.x 接口太旧，Qwen3.5Moe 支持存疑） |

## 坑（实测踩过）

1. **vllm 0.30.x nightly 是 cu13 坏件**：装过 0.30.0 / 0.30.1rc1 的 env 必须
   `pip uninstall -y vllm` 清干净再装 0.19.1，残留会与新 torch ABI 冲突。
2. **换 torch 必须连带 nvidia-* 包整体重装**：混装出过
   `undefined symbol: ncclCommResume`（nccl/cuda 库版本不配）。遇到这类
   undefined symbol，别逐个试，直接重建 env 最省时间。
3. **PyPI pin 对照**（省得再查）：vllm 0.18.0–0.19.1 → torch 2.10.0；vllm
   0.20+ → torch 2.11/2.13（在这台机器上是死路）。
4. **共享机器红线**：一切装在 `~/anaconda3/envs/` 下，用户态隔离；不碰
   `/usr`、不碰内核驱动、不动别人的环境（AGENTS.md 红线 5 同类原则）。

## 相关 env 分工

| env | 用途 |
|---|---|
| `vllm_py310` | vLLM serving / v8 插件 / triton microbench（本文件） |
| `lut_py310` | HF transformers 侧实验（v6 评估、integration 测试），torch 2.6 cu124 |
