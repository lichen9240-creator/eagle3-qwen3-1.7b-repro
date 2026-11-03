# Qwen3-1.7B + EAGLE-3 复现（v1）

在单张 AutoDL RTX 4090 上，从零跑通 **Qwen3-1.7B 的 EAGLE-3 投机解码**完整流水线：
ShareGPT 采样 → target 重新生成 → online 训练 1 层 draft → 导出 → SGLang 部署 → 与 baseline 对比测速。

本仓库记录 **v1（30k 数据）** 的完整执行过程、脚本、门禁结果与实测数据。

---

## 核心结果

| 指标 | 结果 |
|---|---|
| acceptance length (C=1) | **2.090** |
| 吞吐 C=1 | 230.4 → **256.9** tok/s（**1.115x**） |
| 吞吐 C=4 | 781.4 → 843.3 tok/s（1.079x） |
| 吞吐 C=8 | 1464.0 → 1478.7 tok/s（1.010x） |
| 输出一致性（10 条 greedy，64 token） | **5/10**（原因见下） |

C=1 延迟：TPOT 4.22 ms → 3.80 ms，TTFT 19.27 ms → 15.45 ms。

> 结论：**方案的两项硬门禁（acceptance ≥2.0、speedup ≥1.0）都过了，但没拿到 2x 加速。**

---

## 关键洞察：为什么 2.0 的 acceptance 只换来 1.1x

用实测数据可以推出一个简单关系：

```
speedup = acceptance / overhead_factor
```

- `overhead_factor` = 每轮验证耗时 ÷ 单 token 前向耗时 = 7.94 ms / 4.22 ms = **1.88**
- 当前 acceptance 2.09 → **1.11x**（与实测吻合）
- **盈亏平衡（1.0x）需要 acceptance ≥ 1.88**
- **2.0x 需要 acceptance ≥ 3.76**

对 1.7B 这种小 target，验证 4 个位置的前向并不比 1 个便宜多少（decode 是 memory-bound），
再叠加 3 步 draft 前向，overhead 就固定在 ~1.9 倍。**所以在"3 步 chain draft + topk=1"的配置下，
必须把 acceptance 做到 3.8 以上才可能 2x。**

---

## 环境

| 项目 | 值 |
|---|---|
| GPU | RTX 4090 48GB（AutoDL Pro，sm_89，450W） |
| 驱动 / CUDA | 580.105.08 / CUDA 12.8 |
| Python | 3.12.16（conda env） |
| PyTorch | 2.9.1+cu128 |
| Transformers | 4.57.1 |
| SGLang | 0.5.9 |
| SpecForge | `961ca7c3bb9d1d0a7aaf44db35e76e1f227a7be0` |

---

## 目录结构

```
configs/                  draft config、chat 模板、benchmark prompt
docs/                     复现报告、执行记录（含全部偏差登记）
scripts/                  SSH/监控/抓卡等工具脚本
scripts/pipeline/         按门禁顺序编号的流水线脚本（00 → 71）
results/v1/               baseline 与 EAGLE 的 summary、correctness、训练曲线
data_samples/             各类数据的前 3 条样例（截断）
```

---

## 流水线（scripts/pipeline）

| 编号 | 作用 |
|---|---|
| `00_recon.sh` / `01_network.sh` | 环境/网络侦察 |
| `02_env_clone.sh` | 建 Python 3.12 环境 + clone SpecForge 锁 commit |
| `03_install.sh` / `03b_model.sh` | 装依赖 / 下 Qwen3-1.7B |
| `04_data.sh` → `04c_verify.sh` | ShareGPT 下载 → 去重+结构过滤 → 30k 切分 → 校验 |
| `07_regen_server.sh` | 起 SGLang 生成服务（no-think 模板） |
| `08_regen_smoke.sh` | 100 条 smoke |
| `10_full_regen.sh` | 28.5k + 1.5k 全量 regeneration |
| `11_filter.sh` | zero-loss / 2048 过滤 |
| `12_config_audit.sh` | draft config 审计 |
| `13_hidden_schema.sh` | hidden state schema 校验 |
| `14_smoke_train.sh` → `15_train_1k.sh` | 20 步 smoke → 1k 小训 |
| `17_train_30k.sh` | 正式训练（3 epochs） |
| `18_export.sh` | 导出 SGLang 可加载格式 |
| `36/39/37/40` | baseline 服务与测速、EAGLE 服务与测速 |
| `44_determinism.sh` | 输出一致性/确定性验证 |

---

## 踩过的坑（都已解决，脚本里有对应修复）

**环境**
1. 镜像自带 `nvcc` 是 CUDA 11.8，不支持 `-std=c++20` → 从 NVIDIA redistrib 拉 `cuda_nvcc 12.8.93` 自建 CUDA_HOME。
2. `nvidia-cuda-nvcc-cu12`(pip) 12.x 只含 `ptxas`；conda nvidia 频道的 `cuda-nvcc` 是 17KB 占位包 —— 都不是 nvcc。
3. 系统 gcc 9 缺 `<concepts>` → apt 装 gcc-10，nvcc 用 `-ccbin` 指定。
4. `tvm_ffi` 的 JIT 链接固定找 `$CUDA_HOME/lib64`，redistrib 解到 `lib/` → 建软链。
5. `flex_attention` 的 Triton 模板需要 104KB 共享内存 > sm_89 的 99KB → 改用 `sdpa`。

**数据**
6. ShareGPT 120,675 条里有 **52,052 条重复 id** → 去重。
7. 其中 **35.9% 的对话以 assistant 开头**，regeneration 直接报错 → 在候选池里先剔除结构无效对话。
8. `filter_zero_loss` 在 2048 截断后仍会丢掉约 24%（长对话只剩 user 部分）→ 30k 抽样最终只有 21,750 可训练。

**一致性**
9. G10 要求的"10/10 逐字一致"在 bf16 下基本不可能：EAGLE 自身两次运行 10/10 一致，
   但与 baseline 有 5/10 不同 → 这是 batch 验证 vs 单 token decode 的 kernel 路径差异，不是采样随机性。

---

## 说明

- 模型权重、完整数据集、checkpoint **未上传**（体积原因）。数据可用 `docs/执行记录` 中的流程重建。
- `data_samples/` 只保留每类数据的前 3 条、内容截断到 300 字符，用于说明 schema。
- 本仓库为 **v1（30k 数据）**。v2（41.4k 数据，ShareGPT 30k + UltraChat 13.4k）正在训练中。

## License

仅供学习与研究使用。ShareGPT / UltraChat 数据遵循其原始许可。
