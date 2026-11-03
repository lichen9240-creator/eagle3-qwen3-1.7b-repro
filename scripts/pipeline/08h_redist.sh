set -e
ROOT=/root/autodl-tmp/eagle3-qwen
PREFIX=$ROOT/envs/cuda128
mkdir -p $PREFIX
echo "[$(date +%T)] probe redistrib metadata"
rm -f /tmp/redistrib_ver
for v in 12.8.1 12.8.0; do
  code=$(curl -sL -o /tmp/redistrib_$v.json -w '%{http_code}' --max-time 30 \
    https://developer.download.nvidia.com/compute/cuda/redist/redistrib_$v.json)
  echo "  $v -> $code"
  if [ "$code" = "200" ]; then echo $v > /tmp/redistrib_ver; break; fi
done
[ -f /tmp/redistrib_ver ] || { echo "REDISTRIB_UNREACHABLE"; exit 1; }
V=$(cat /tmp/redistrib_ver)
echo "use CUDA $V"

$ROOT/envs/eagle3/bin/python - "$V" <<'PY'
import json, sys, urllib.request, os
v = sys.argv[1]
j = json.load(open(f"/tmp/redistrib_{v}.json"))
base = "https://developer.download.nvidia.com/compute/cuda/redist/"
want = ["cuda_nvcc", "cuda_cudart", "cuda_cccl"]
os.makedirs("/tmp/cudaredist", exist_ok=True)
for name in want:
    comp = j.get(name, {}).get("linux-x86_64", {})
    rel = comp.get("relative_path")
    if not rel:
        print("MISS", name); continue
    dst = "/tmp/cudaredist/" + os.path.basename(rel)
    if os.path.exists(dst) and os.path.getsize(dst) > 1000:
        print("cached", dst); continue
    print("download", base + rel, flush=True)
    urllib.request.urlretrieve(base + rel, dst)
    print("  ->", os.path.getsize(dst) / 1e6, "MB")
PY

echo "[$(date +%T)] extract to $PREFIX"
for f in /tmp/cudaredist/*.tar.xz; do
  echo "  extract $(basename $f)"
  tar -xJf "$f" -C $PREFIX --strip-components=1 2>/dev/null || tar -xJf "$f" -C $PREFIX
done
echo "[$(date +%T)] verify"
export CUDA_HOME=$PREFIX
ls $CUDA_HOME/bin 2>/dev/null | grep -E 'nvcc|ptxas' || echo "NO_BIN"
$CUDA_HOME/bin/nvcc --version 2>&1 | tail -2
cd /tmp && printf '#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n__global__ void k(){__half h;}\n' > t.cu
$CUDA_HOME/bin/nvcc -std=c++20 -arch=sm_89 -c t.cu -o t.o && echo "CUDA128_NVCC_OK"
