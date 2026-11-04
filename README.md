# Qwen3-1.7B + EAGLE-3 复现

在单张 AutoDL RTX 4090 上，从零跑通 **Qwen3-1.7B 的 EAGLE-3 投机解码**完整流水线：
ShareGPT / UltraChat 采样 → target 重新生成 → online 训练 1 层 draft → 导出 → SGLang 部署 → 与 baseline 对比测速。

本仓库包含**两轮训练（v1/v2）+ 两种推理引擎（SGLang/vLLM）+ 三种 draft 形态（chain k=2/3/4、tree）**的完整执行记录、脚本、门禁结果与实测数据。

> **完整实验报告（推荐先读）：[docs/完整实验报告.md](docs/完整实验报告.md)**
> 涵盖数据构建全过程、训练曲线、SGLang/vLLM 对照、与腾讯 AngelSlim 官方结果的逐项比较。

| 版本 | 训练数据 | 推理引擎 | acceptance (C=1) | 吞吐 C=1 | speedup |
|---|---|---|---|---|---|
| **v1**（我们训的） | 21,750 | SGLang 0.5.9 | 2.090 | 230.4 → 256.9 tok/s | **1.115x** |
| **v2**（我们训的） | 41,405 | SGLang 0.5.9 | 2.268 | 234.0 → 283.6 tok/s | **1.212x** |
| v2 + tree draft（我们训的） | 41,405 | SGLang 0.5.9 | 2.500 | 173.4 → 229.6 tok/s | **1.324x** |
| v2（我们训的） | 41,405 | **vLLM 0.11.2**，n=2 | 2.33 | 152.3 → 242.1 tok/s | **1.590x** |
| v2，1024 token（我们训的） | 41,405 | **vLLM 0.11.2**，n=2 | 2.33 | 150.3 → 282.4 tok/s | **1.879x** |
| v2，1024 token（我们训的） | 41,405 | **vLLM 0.11.2**，n=3 | 2.72 | 150.3 → 308.3 tok/s | **2.051x** |
| **v2，1024 token（我们训的）** | 41,405 | **vLLM 0.11.2**，n=4 | **2.89** | 150.3 → **321.0** tok/s | **2.135x** 🎯 |
| AngelSlim 开源权重（参考） | — | vLLM 0.11.2，n=2 | 2.30 | 150.3 → 264.7 tok/s | 1.761x |

> **🎯 2x 已达成**：`vLLM + 1024 token + num_speculative_tokens=4`，在同一张消费级 4080S 上做到 **2.135x**。
> 这不是调参运气——acceptance 2.89、O 1.354，两者都在实测曲线上。
>
> **同一张 4080S、同一套 vLLM 配置、同一 benchmark，我们自训的 v2 draft 比腾讯开源的 AngelSlim 快约 6.7%**
> （1024 token，同为 n=2：282.4 vs 264.7 tok/s），acceptance 也持平略高（2.33 vs 2.30）。
>
> 与官方公布的 1.687x 相比：**不是同一评测集**（他们用 GSM8K/Alpaca/HumanEval/MT-bench 平均，
> 我们用 90 条自建 prompt），只能作为量级参考。

**🤗 模型已发布到 HuggingFace：[Lichen-Creator/Qwen3-1.7B_eagle3](https://huggingface.co/Lichen-Creator/Qwen3-1.7B_eagle3)**
（v2 draft，可直接被 SGLang 0.5.9 / vLLM 0.11.2 加载）

---

## 文档索引

| 文档 | 内容 |
|---|---|
| [docs/完整实验报告.md](docs/完整实验报告.md) | **主报告**：数据构建、训练、SGLang/vLLM 对照、与 AngelSlim 比较 |
| [docs/2x不可达性分析.md](docs/2x不可达性分析.md) | 加速比上限的数学分析（含对"2x 不可达"的自我更正） |
| [docs/vLLM三向对照.md](docs/vLLM三向对照.md) | 我们 vs AngelSlim vs baseline 的同卡对照 |
| **[docs/DualDraft-实验报告.md](docs/DualDraft-实验报告.md)** | **衍生实验：GtoA/AtoA 双网络（负结果）+ 损失衰减率扫描** |
| [docs/复现报告-v1.md](docs/复现报告-v1.md) / [-v2.md](docs/复现报告-v2.md) | 两轮训练的专项报告 |
| [docs/实验执行记录.md](docs/实验执行记录.md) | 逐项偏差登记 |

### 衍生实验一句话总结

把 EAGLE-3 的 draft 拆成 GtoA + AtoA 两个独立网络**是亏的**（同权重下 τ −0.075，且多 85% 参数）；
但同一实验发现**损失衰减率被社区两套实现设成了不同值**（EAGLE `0.8^i` vs vLLM Speculators `1.0`），
实测最优点在 `0.7^i`（step0 占 32.7%），相对 `0.8^i` 提升 **+0.105**。
详见 [DualDraft 报告](docs/DualDraft-实验报告.md)。


---

## 关键洞察：加速比到底由什么决定（完整推理链）

把加速比拆成两个可测量的量：

```
speedup S = τ / O
τ = acceptance（每轮接受的 token 数）
O = 每轮验证耗时 ÷ 单 token 前向耗时
O  = 1 + k·[(1-r) + δ],   δ = W_draft / W_target（draft 与 target 的权重比）
```

下面是完整的推理链，**包括一个被自己的实验推翻的假设**：

| 步骤 | 做法 | 结果 |
|---|---|---|
| ① 观察 | v2 训完，acceptance 2.268 却只有 1.212x | 差距不在 acceptance |
| ② 建模 | 反解 O = τ/S = 2.268/1.212 | **O = 1.87**（SGLang） |
| ③ 假设 | 假设 O 是结构常数 → 2x 需 acceptance ≥ 3.74 | 判定"2x 不可达" |
| ④ 证伪 | 换引擎（同卡、同 draft、同 benchmark） | **O = 1.24**，假设被推翻 |
| ⑤ 重算 | 2x 只需 acceptance ≥ 2.48 | 与本已测到的 2.50(tree) 只差一步 |
| ⑥ 验证 | vLLM + 1024 token + n=4 | **2.135x，2x 达成** |
| ⑦ 边界 | 换回 128 token 输出 | 上限只有 1.676x —— 2x 依赖长生成 |

**关键教训**：`O` 不是硬件常数，而是**推理栈实现**的函数。
vLLM 对 EAGLE draft head 做了 torch.compile + CUDA graph，而 SGLang 0.5.9 的 spec 路径是 eager，
每步多出约 0.5 ms 固定开销——draft 真实计算只有 274MB/717GB/s ≈ 0.38 ms。
把这 0.5 ms 压掉，O 就从 1.87 掉到 1.24，2x 的门槛从 3.74 降到 2.48。

> 换句话说：**先选对推理栈，再谈调参。** 顺序反了会得出"2x 不可能"的错误结论——我们一开始就是这么错的，
> 完整复盘见 [docs/2x不可达性分析.md](docs/2x不可达性分析.md)（该文保留了错误推理的原文 + 第 10 节的更正）。

---

## 实验矩阵

| draft | 引擎 | 参数 | 输出长度 | 并发 | 结果 |
|---|---|---|---|---|---|
| v1（21.75k） | SGLang 0.5.9 | k=3 | 128 | 1/4/8 | 1.115x |
| v2（41.4k） | SGLang 0.5.9 | k=3 | 128 | 1/4/8 | 1.212x |
| v2 | SGLang 0.5.9 | k=2 / 3 / **4** | 128 | 1 | 1.154 / **1.219** / 1.155x |
| v2 | SGLang 0.5.9 | tree topk=2，8 节点 | 128 | 1 | **1.324x** |
| AngelSlim | SGLang 0.5.9 | k=2 / 3 | 128 | 1 | 1.116 / 1.149x |
| AngelSlim | vLLM 0.11.2 | n=2 | 128 / 1024 | 1, 4 | 1.534 / 1.761x |
| **v2** | **vLLM 0.11.2** | n=2 | 128 / 1024 | 1, 4 | **1.590 / 1.879x** |
| **v2** | **vLLM 0.11.2** | n=3 | 1024 | 1 | **2.051x** |
| **v2** | **vLLM 0.11.2** | **n=4** | **1024** | 1 | **🎯 2.135x** |

共 **9 组 draft×引擎×参数 组合**、**5 种输出/并发配置**，覆盖 chain 与 tree 两种 draft 形态。

**完整结果**

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

数据包在 [Releases](https://github.com/lichen9240-creator/eagle3-qwen3-1.7b-repro/releases/tag/data-v1-v2)：
- `eagle3-qwen3-1.7b-v1-data.tar.gz`（train 21,750 + eval 1,154 + bench_90）
- `eagle3-qwen3-1.7b-v2-data.tar.gz`（train 41,405 + eval 2,179 + bench_90）

---

## 环境

| 项目 | 值 |
|---|---|
| 训练 GPU | RTX 4090 48GB（AutoDL Pro，sm_89，450W） |
| 驱动 / CUDA | 580.105.08 / CUDA 12.8 |
| Python | 3.12.16 |
| PyTorch | 2.9.1+cu128 |
| Transformers | 4.57.1 |
| 推理引擎 | SGLang 0.5.9 / vLLM 0.11.2 |
| SpecForge | `961ca7c3bb9d1d0a7aaf44db35e76e1f227a7be0` |
| 复测 GPU | RTX 4080 SUPER 32GB（vLLM 对照 + k 扫描 + tree） |

---

## 目录结构

```
REPRODUCE.md              从零复现的详细手册
configs/                  draft config、chat 模板、benchmark prompt
docs/                     完整实验报告、2x 分析、v1 / v2 报告、vLLM 三向对照、执行记录
scripts/                  SSH/监控等工具
scripts/pipeline/         按门禁编号的流水线脚本（00 → 110，含 vLLM 对照）
results/v1 v2            4090 上的 summary、correctness、训练曲线
results/ksweep_sglang/    k=2/3/4 扫描（含 baseline）
results/treesweep_sglang/ tree topk=2、8 节点
results/vllm vllm_ours vllm_tree   vLLM 对照（AngelSlim / 我们 n=2 / 我们 n=3,4）
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

- 模型权重已发布到 [HuggingFace](https://huggingface.co/Lichen-Creator/Qwen3-1.7B_eagle3)，也可从 [Releases](https://github.com/lichen9240-creator/eagle3-qwen3-1.7b-repro/releases/tag/data-v1-v2) 下载 tar 包。
- `data_samples/` 只保留每类数据的前 3 条、内容截断到 300 字符，用于说明 schema。
- ShareGPT / UltraChat 数据遵循其原始许可。

## License

仅供学习与研究使用。
