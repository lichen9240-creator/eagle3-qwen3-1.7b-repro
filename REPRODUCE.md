# 复现手册（从零到 benchmark）

> 适用于：单卡 RTX 4090（24G 或 48G）、AutoDL 类似环境。
> 本文按我们实际跑通的顺序写，包含所有踩过的坑的修复方法。
> 预计总耗时：**约 16 小时**（数据 3h + 训练 11h + 评测 0.5h），成本约 ¥45（按 ¥2.88/h）。

---

## 0. 准备

```bash
# 宿主机要求：GPU 驱动 >= 570（CUDA 12.8），磁盘 >= 60GB
nvidia-smi
df -h /root/autodl-tmp
```

```bash
export ROOT=/root/autodl-tmp/eagle3-qwen
mkdir -p $ROOT/{models,data,runs,exports,logs,cache,envs}
cd $ROOT
```

---

## 1. 环境（最容易卡住的一步）

镜像自带的 `nvcc` 是 CUDA 11.8，而 sglang 的 JIT 内核需要 **CUDA 12 + gcc-10**。按顺序做：

### 1.1 Python 3.12

```bash
export PATH=/root/miniconda3/bin:$PATH
conda create -y -p $ROOT/envs/eagle3 python=3.12
export PATH=$ROOT/envs/eagle3/bin:$PATH
```

### 1.2 自建 CUDA 12.8 工具链

> pip 的 `nvidia-cuda-nvcc-cu12` 只含 `ptxas`，conda 的 `cuda-nvcc` 是 17KB 占位包，都不可用。

```bash
V=12.8.1
curl -sL -o /tmp/redistrib.json \
  https://developer.download.nvidia.com/compute/cuda/redist/redistrib_$V.json

python - <<'PY'
import json, os, urllib.request
j = json.load(open("/tmp/redistrib.json"))
base = "https://developer.download.nvidia.com/compute/cuda/redist/"
os.makedirs("/tmp/cudaredist", exist_ok=True)
for name in ["cuda_nvcc", "cuda_cudart", "cuda_cccl"]:
    rel = j[name]["linux-x86_64"]["relative_path"]
    dst = "/tmp/cudaredist/" + os.path.basename(rel)
    urllib.request.urlretrieve(base + rel, dst)
    print("got", dst)
PY

mkdir -p $ROOT/envs/cuda128
for f in /tmp/cudaredist/*.tar.xz; do
  tar -xJf "$f" -C $ROOT/envs/cuda128 --strip-components=1
done
export CUDA_HOME=$ROOT/envs/cuda128
export PATH=$CUDA_HOME/bin:$PATH
$CUDA_HOME/bin/nvcc --version      # 应显示 12.8
```

### 1.3 gcc-10（C++20 `<concepts>`）

```bash
apt-get update -qq && apt-get install -y gcc-10 g++-10
mkdir -p $ROOT/envs/hostbin
for n in gcc g++ cc c++; do :; done
ln -sf /usr/bin/gcc-10  $ROOT/envs/hostbin/gcc
ln -sf /usr/bin/g++-10  $ROOT/envs/hostbin/g++
ln -sf /usr/bin/gcc-10  $ROOT/envs/hostbin/cc
ln -sf /usr/bin/g++-10  $ROOT/envs/hostbin/c++
export PATH=$ROOT/envs/hostbin:$PATH
export CC=$ROOT/envs/hostbin/gcc CXX=$ROOT/envs/hostbin/g++
export NVCC_PREPEND_FLAGS="-ccbin $ROOT/envs/hostbin/g++"
# tvm_ffi 的 JIT 链接固定找 lib64
ln -sfn $CUDA_HOME/lib $CUDA_HOME/lib64
```

### 1.4 装 PyTorch 与 SpecForge

> 别用国内 PyPI 镜像装 torch（实测 15KB/s），用官方 CDN 或 ustc。

```bash
python -m pip install -U pip setuptools wheel
python -m pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu128

git clone https://github.com/sgl-project/SpecForge.git $ROOT/SpecForge
cd $ROOT/SpecForge
git checkout 961ca7c3bb9d1d0a7aaf44db35e76e1f227a7be0
python -m pip install -e . -i https://mirrors.ustc.edu.cn/pypi/simple \
  --trusted-host mirrors.ustc.edu.cn
```

验收：

```bash
python - <<'PY'
import torch, transformers, sglang
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print(transformers.__version__, sglang.__version__)   # 4.57.1 / 0.5.9
PY
```

### 1.5 下载 target 模型

```bash
export HF_ENDPOINT=https://hf-mirror.com
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-1.7B', local_dir='$ROOT/models/Qwen3-1.7B',
                  allow_patterns=['*.json','*.safetensors','*.txt','*.jinja','merges.txt','vocab.json'])
"
```

---

## 2. 数据

### 2.1 ShareGPT 30k

```bash
cd $ROOT/SpecForge
python scripts/prepare_data.py --dataset sharegpt --output-path $ROOT/data/sharegpt_full
# → 120,675 条（其中 52,052 条 id 重复，35.9% 以 assistant 开头）
```

用本仓库 `scripts/pipeline/04b_split.sh` 做去重 + 结构过滤 + seed 42 抽样：

```bash
python repro/scripts/split_sharegpt.py \
  --input $ROOT/data/sharegpt_full/sharegpt_train.jsonl \
  --output-dir $ROOT/data/sharegpt_30k
# → 28,500 / 1,500
```

### 2.2 启动生成服务

```bash
nohup python -m sglang.launch_server \
  --model-path $ROOT/models/Qwen3-1.7B --served-model-name Qwen3-1.7B \
  --chat-template repro/templates/qwen3_nothink.jinja \
  --dtype bfloat16 --mem-fraction-static 0.70 --max-running-requests 64 \
  --random-seed 42 --host 0.0.0.0 --port 30000 > $ROOT/logs/regen_server.log 2>&1 &
# 等 curl http://127.0.0.1:30000/health 返回 200
```

> ⚠️ 一定要确认端口没被占：`grep 'address already in use' $ROOT/logs/regen_server.log` 应为空。
> 我们踩过：旧的 EAGLE3 服务占着端口，新服务绑定失败，regeneration 实际跑在投机解码上。

### 2.3 重新生成回答

```bash
cd $ROOT/SpecForge
for split in train eval; do
python scripts/regenerate_train_data.py \
  --model Qwen3-1.7B --concurrency 32 --max-tokens 512 \
  --temperature 0.8 --top-p 0.95 --top-k 20 \
  --server-address 127.0.0.1:30000 \
  --input-file-path $ROOT/data/sharegpt_30k/sharegpt_30k_${split}.jsonl \
  --output-file-path $ROOT/data/regen/sharegpt_30k_${split}_regen.jsonl
done
# 约 2.2 条/秒，30k 需要 ~3.6 小时
```

### 2.4 过滤

```bash
for split in train eval; do
python repro/scripts/filter_zero_loss.py \
  --model $ROOT/models/Qwen3-1.7B \
  --input $ROOT/data/regen/sharegpt_30k_${split}_regen.jsonl \
  --output $ROOT/data/regen/${split}_filtered.jsonl --max-length 2048
done
# 28,500 → 21,750（保留 76.5%）
```

### 2.5 （可选）扩到 41k：UltraChat

```bash
python scripts/prepare_data.py --dataset ultrachat \
  --output-path $ROOT/data/ultrachat_full --sample-size 35000
# 取前 30,000 条结构有效的，用同一服务重新生成 + 过滤，见 scripts/pipeline/64~68
```

把 ShareGPT 与 UltraChat 的 filtered 数据合并、按 seed 42 洗牌、95/5 切分，得到 v2 的 `41,405 / 2,179`。

### 2.6 固定 benchmark prompt

```bash
python repro/scripts/tokenize_bench.py \
  --model $ROOT/models/Qwen3-1.7B \
  --template repro/templates/qwen3_nothink.jinja \
  --input repro/data/bench_prompts.jsonl \
  --output $ROOT/data/bench/bench_90.jsonl
# 90 条（40 对话 / 30 数学 / 30 代码）
```

---

## 3. 训练 draft

```bash
cd $ROOT/SpecForge
# 注意：attention backend 用 sdpa，flex_attention 在本卡会因共享内存超限失败
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
  scripts/train_eagle3.py \
  --target-model-path $ROOT/models/Qwen3-1.7B \
  --draft-model-config configs/qwen3-1.7b-eagle3.json \
  --train-data-path $ROOT/data/v2/train_v2.jsonl \
  --eval-data-path $ROOT/data/v2/eval_v2.jsonl \
  --output-dir $ROOT/runs/qwen3-1.7b-eagle3-42k \
  --num-epochs 3 --batch-size 1 --learning-rate 1e-4 \
  --max-length 2048 --ttt-length 7 --chat-template qwen3-instruct \
  --target-model-backend sglang --attention-backend sdpa \
  --embedding-key model.embed_tokens.weight --cache-dir $ROOT/cache \
  --save-interval 5000 --eval-interval 10000 --log-interval 100 \
  --seed 42 --report-to none --sglang-mem-fraction-static 0.35 \
  --sglang-context-length 2048 \
  --dataloader-num-workers 4 --build-dataset-num-proc 16
# 41,405 × 3 = 124,215 步，约 11 小时（3.2 步/秒）
```

> **坑**：`--attention-backend flex_attention` 会报
> `No valid triton configs. OutOfMemoryError: triton_tem_fused_0 Required: 106496 Hardware limit: 101376`
> 这是共享内存限制（不是显存），换 `sdpa` 即可，数学等价。

---

## 4. 导出与部署

```bash
CKPT=$(ls -d $ROOT/runs/qwen3-1.7b-eagle3-42k/epoch_* | sort -V | tail -1)
mkdir -p $ROOT/exports/qwen3-1.7b-eagle3-v2-42k
cp $CKPT/config.json $CKPT/model.safetensors $ROOT/exports/qwen3-1.7b-eagle3-v2-42k/
```

启动 EAGLE3 服务：

```bash
python -m sglang.launch_server \
  --model-path $ROOT/models/Qwen3-1.7B \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path $ROOT/exports/qwen3-1.7b-eagle3-v2-42k \
  --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --dtype bfloat16 --mem-fraction-static 0.55 \
  --max-running-requests 8 --disable-radix-cache --host 127.0.0.1 --port 30000
```

日志里应出现：`Load weight end. type=LlamaForCausalLMEagle3`。

---

## 5. 评测

```bash
# baseline（同端口换不带 draft 的服务）
python repro/scripts/check_outputs.py  --data $ROOT/data/bench/bench_90.jsonl \
  --out $ROOT/runs/baseline_v2_correctness.json --n 10 --max-tokens 64
python repro/scripts/bench_serving.py --url http://127.0.0.1:30000/generate \
  --data $ROOT/data/bench/bench_90.jsonl \
  --out-dir $ROOT/runs/baseline_v2 --tag baseline \
  --concurrency 1 4 8 --requests 90 --max-tokens 128 --warmup 3

# EAGLE：同样两条命令，改 --out/--out-dir/--tag
# 然后比对
python repro/scripts/compare_outputs.py \
  --baseline $ROOT/runs/baseline_v2_correctness.json \
  --eagle    $ROOT/runs/eagle_v2_correctness.json
```

固定采样口径：`temperature=0, top_k=1, top_p=1.0, ignore_eos=True`，baseline 与 EAGLE 完全一致。

---

## 6. 预期结果（用于自查）

| 数据量 | acceptance C=1 | 吞吐 C=1 | speedup |
|---|---|---|---|
| 21,750 | 2.09 | 230 → 257 tok/s | 1.115x |
| 41,405 | **2.27** | 234 → 284 tok/s | **1.212x** |

**两个已知现象，不是 bug**：

1. **一致性 5/10**：greedy 输出与 baseline 有差异。已验证 EAGLE 自身两次运行 10/10 一致，
   差异来自 bf16 下 batch 验证与单 token decode 的 kernel 路径不同。
2. **到不了 2x**：`speedup = acceptance / 1.87`，2x 需要 acceptance ≥ 3.74，
   而"3 步 chain draft + topk=1"这个配置下，即使 41k 数据也只到 2.27。

---

## 7. 时间与成本参考

| 阶段 | 时长 | 备注 |
|---|---|---|
| 环境搭建 | 1.5 h | 主要耗在 nvcc/gcc/依赖下载 |
| ShareGPT regen 30k | 3.6 h | 2.2 条/秒 |
| UltraChat regen 15k | 2.3 h | |
| 训练 41.4k × 3 | 11 h | 3.2 步/秒 |
| 评测 | 0.5 h | |

按 AutoDL 4090-48G ¥2.88/h 计，总计约 **¥55**。
