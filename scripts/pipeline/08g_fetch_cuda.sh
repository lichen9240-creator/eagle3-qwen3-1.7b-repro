set -e
ROOT=/root/autodl-tmp/eagle3-qwen
PY=$ROOT/envs/eagle3/bin/python
pkill -f 'conda create' 2>/dev/null || true
$PY -m pip install -q --no-cache-dir zstandard \
  -i https://mirrors.ustc.edu.cn/pypi/simple --trusted-host mirrors.ustc.edu.cn
mkdir -p $ROOT/envs/cuda128
echo "[$(date +%T)] 拉取 nvidia 频道包并解压"
$PY - <<'PY'
import json, os, tarfile, urllib.request, zipfile, io
import zstandard as zstd

BASE = "https://conda.anaconda.org/nvidia/linux-64/"
PREFIX = "/root/autodl-tmp/eagle3-qwen/envs/cuda128"
WORK = "/tmp/cudapkgs"
os.makedirs(WORK, exist_ok=True)

print("fetch repodata ...", flush=True)
rp = json.load(urllib.request.urlopen(BASE + "repodata.json", timeout=180))
pool = {**rp.get("packages", {}), **rp.get("packages.conda", {})}
print("packages in repo:", len(pool))

targets = {"cuda-nvcc", "cuda-cudart", "cuda-cudart-dev", "cuda-cccl"}
best = {}
for fn, m in pool.items():
    if m.get("name") not in targets:
        continue
    if not m.get("version", "").startswith("12.8"):
        continue
    key = m["name"]
    cand = (int(m.get("build_number", 0)), fn, m)
    if key not in best or cand[0] > best[key][0]:
        best[key] = cand

assert "cuda-nvcc" in best, "cuda-nvcc 12.8 not found"
for k, (_, fn, m) in sorted(best.items()):
    print(f"  pick {k}: {fn} ({m.get('size',0)/1e6:.0f}MB)")

def extract_conda(path):
    if path.endswith(".tar.bz2"):
        with tarfile.open(path, "r:bz2") as tf:
            tf.extractall(PREFIX)
        return
    with zipfile.ZipFile(path) as z:
        z.extractall(WORK + "/_x")
    for root, _, files in os.walk(WORK + "/_x"):
        for f in files:
            if f.endswith(".tar.zst"):
                p = os.path.join(root, f)
                with open(p, "rb") as fh:
                    dctx = zstd.ZstdDecompressor()
                    with dctx.stream_reader(fh) as r:
                        with tarfile.open(fileobj=r, mode="r|") as tf:
                            tf.extractall(PREFIX)

for k, (_, fn, m) in sorted(best.items()):
    dst = os.path.join(WORK, fn)
    if not os.path.exists(dst) or os.path.getsize(dst) != m.get("size", -1):
        print("download", fn, flush=True)
        urllib.request.urlretrieve(BASE + fn, dst)
    print("extract", fn, flush=True)
    extract_conda(dst)

print("done")
PY
echo "[$(date +%T)] 验证 nvcc"
export CUDA_HOME=$ROOT/envs/cuda128
export PATH=$CUDA_HOME/bin:$PATH
ls $CUDA_HOME/bin | grep -E '^nvcc|^ptxas' || true
$CUDA_HOME/bin/nvcc --version 2>&1 | tail -2
echo "=== 编译测试 ==="
cd /tmp && printf '#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n__global__ void k(){__half h;}\n' > t.cu
$CUDA_HOME/bin/nvcc -std=c++20 -arch=sm_89 -c t.cu -o t.o && echo "CUDA128_NVCC_OK"
