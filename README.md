# Qwen3-1.7B + EAGLE-3 复现

在单张 AutoDL RTX 4090 上，从零跑通 **Qwen3-1.7B 的 EAGLE-3 投机解码**完整流水线：
ShareGPT / UltraChat 采样 → target 重新生成 → online 训练 1 层 draft → 导出 → SGLang 部署 → 与 baseline 对比测速。

本仓库包含**两轮实验**的完整执行记录、脚本、门禁结果与实测数据：

| 版本 | 训练数据 | acceptance (C=1) | 吞吐 C=1 | speedup |
|---|---|---|---|---|
| **v1** | 21,750 | 2.090 | 230.4 → 256.9 tok/s | **1.115x** |
| **v2** | 41,405 | **2.268** | 234.0 → 283.6 tok/s | **1.212x** |

> **结论：验收门禁（acceptance ≥2.0、speedup ≥1.0）通过，但 2x 未达到——而且这不是"数据不够"。**

---

## 关键洞察：2x 需要什么

用实测数据可以推出：

```
speedup = acceptance / overhead_factor
overhead_factor = 每轮验证耗时 ÷ 单 token 前向耗时 ≈ 1.87   （v1/v2 都是这个值）
```

- acceptance 2.27 → 1.21x（与实测吻合）
- **盈亏平衡（1.0x）需要 acceptance ≥ 1.87**
- **2.0x 需要 acceptance ≥ 3.74**

对 1.7B 这种小 target，验证 4 个位置的前向并不比 1 个便宜多少（decode 是 memory-bound），
再叠加 3 步 draft 前向，overhead 就固定在 ~1.87 倍。数据从 21.75k 翻倍到 41.4k，
acceptance 只涨 0.18——**瓶颈不在数据，在推理配置的 overhead 结构**。

---

## 完整结果

### v2（41,405 条）vs v1（21,750 条）

| 指标 | v1 | v2 |
|---|---|---|
| acceptance (C=1 / C=4 / C=8) | 2.090 / 2.089 / 2.097 | **2.268 / 2.255 / 2.256** |
| 吞吐 C=1 | 1.115x | **1.212x** |
| 吞吐 C=4 | 1.079x | **1.207x** |
| 吞吐 C=8 | 1.010x | **1.190x** |
| TPOT | 4.22 → 3.80 ms | 4.18 → 3.44 ms |
| 训练末 loss | 0.87 | **0.60** |
| 输出一致性 | 5/10 | 5/10 |

### 关于输出一致性 5/10

10 条 greedy prompt、64 token，baseline 与 EAGLE 有 5 条不同，但：
**EAGLE 自身两次运行 10/10 逐字一致**。差异来自 bf16 下"4 位置批量验证"与"单 token decode"
的 GEMM 形状不同导致的 logits 微差，在近似平票处 argmax 翻转（分歧都在第 40–60 token 之后）。
不是采样随机性，也与训练量无关。详见 `docs/复现报告-v1.md` §4。

---

## 快速开始

**详细复现手册见 [REPRODUCE.md](REPRODUCE.md)**（含所有环境坑的修复方法、逐条命令、预期结果）。

数据包在 [Releases](../../releases)：
- `eagle3-qwen3-1.7b-v1-data.tar.gz`（train 21,750 + eval 1,154 + bench_90）
- `eagle3-qwen3-1.7b-v2-data.tar.gz`（train 41,405 + eval 2,179 + bench_90）

---

## 环境

| 项目 | 值 |
|---|---|
| GPU | RTX 4090 48GB（AutoDL Pro，sm_89，450W） |
| 驱动 / CUDA | 580.105.08 / CUDA 12.8 |
| Python | 3.12.16 |
| PyTorch | 2.9.1+cu128 |
| Transformers | 4.57.1 |
| SGLang | 0.5.9 |
| SpecForge | `961ca7c3bb9d1d0a7aaf44db35e76e1f227a7be0` |

---

## 目录结构

```
REPRODUCE.md              从零复现的详细手册
configs/                  draft config、chat 模板、benchmark prompt
docs/                     复现报告 v1 / v2、执行记录（含全部偏差登记）
scripts/                  SSH/监控等工具
scripts/pipeline/         按门禁编号的流水线脚本（00 → 78）
results/v1/ results/v2/   summary、correctness、训练曲线
data_samples/             各类数据的前 3 条样例（截断）
```

---

## 踩过的坑（都已解决，脚本里有对应修复）

**环境**
1. 镜像自带 `nvcc` 是 CUDA 11.8，不支持 `-std=c++20` → 从 NVIDIA redistrib 拉 `cuda_nvcc 12.8` 自建 CUDA_HOME。
2. `nvidia-cuda-nvcc-cu12`(pip) 12.x 只含 `ptxas`；conda nvidia 频道的 `cuda-nvcc` 是 17KB 占位包。
3. 系统 gcc 9 缺 `<concepts>` → apt 装 gcc-10，nvcc 用 `-ccbin` 指定。
4. `tvm_ffi` 的 JIT 链接固定找 `$CUDA_HOME/lib64`，redistrib 解到 `lib/` → 建软链。
5. `flex_attention` 的 Triton 模板需要 104KB 共享内存 > sm_89 的 99KB → 改用 `sdpa`。

**数据**
6. ShareGPT 120,675 条里有 **52,052 条重复 id** → 去重。
7. 其中 **35.9% 的对话以 assistant 开头**，regeneration 直接报错 → 在候选池里先剔除结构无效对话。
8. `filter_zero_loss` 在 2048 截断后仍会丢掉约 24% → 30k 抽样最终只有 21,750 可训练。
9. 两个实例各自抽的 30k 样本**重叠 68.6%**（同池同 seed）→ 合并只有 39,426 条唯一对话。

**一致性**
10. 要求的"10/10 逐字一致"在 bf16 下不可能满足（见上）。

---

## 说明

- 模型权重、checkpoint 未上传（体积原因），可由 REPRODUCE.md 重新训练得到。
- `data_samples/` 只保留每类数据的前 3 条、内容截断到 300 字符，用于说明 schema。
- ShareGPT / UltraChat 数据遵循其原始许可。

## License

仅供学习与研究使用。
