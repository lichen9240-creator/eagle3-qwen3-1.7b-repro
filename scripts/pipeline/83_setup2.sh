set -e
ROOT=/root/autodl-tmp/eagle3-qwen
pkill -9 -f 'pip install' || true
sleep 3
rm -rf /tmp/pip-* 2>/dev/null || true
export PATH=$ROOT/envs/eagle3/bin:$PATH

echo "[$(date +%T)] === 2. torch (bfsu 源) ==="
python -m pip install --no-cache-dir torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
  -i https://mirrors.bfsu.edu.cn/pypi/web/simple \
  --trusted-host mirrors.bfsu.edu.cn 2>&1 | tail -3
python -c "import torch;print('torch',torch.__version__,torch.version.cuda,torch.cuda.is_available(),torch.cuda.get_device_name(0))"

echo "[$(date +%T)] === 3. CUDA 12.8 nvcc ==="
mkdir -p $ROOT/envs/cuda128
curl -sL -o /tmp/redistrib.json https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.8.1.json
python - <<'PY'
import json, os, urllib.request
j = json.load(open("/tmp/redistrib.json"))
base = "https://developer.download.nvidia.com/compute/cuda/redist/"
os.makedirs("/tmp/cudaredist", exist_ok=True)
for name in ["cuda_nvcc", "cuda_cudart", "cuda_cccl"]:
    rel = j[name]["linux-x86_64"]["relative_path"]
    dst = "/tmp/cudaredist/" + os.path.basename(rel)
    if not os.path.exists(dst):
        urllib.request.urlretrieve(base + rel, dst)
    print("ok", os.path.basename(dst))
PY
for f in /tmp/cudaredist/*.tar.xz; do tar -xJf "$f" -C $ROOT/envs/cuda128 --strip-components=1; done
export CUDA_HOME=$ROOT/envs/cuda128
ln -sfn $CUDA_HOME/lib $CUDA_HOME/lib64
$CUDA_HOME/bin/nvcc --version | tail -1

echo "[$(date +%T)] === 4. gcc-10 ==="
apt-get install -y -qq gcc-10 g++-10 2>&1 | tail -2
mkdir -p $ROOT/envs/hostbin
ln -sf /usr/bin/gcc-10 $ROOT/envs/hostbin/gcc
ln -sf /usr/bin/g++-10 $ROOT/envs/hostbin/g++
ln -sf /usr/bin/gcc-10 $ROOT/envs/hostbin/cc
ln -sf /usr/bin/g++-10 $ROOT/envs/hostbin/c++
$ROOT/envs/hostbin/g++ --version | head -1

echo "[$(date +%T)] === 5. SpecForge ==="
source /etc/network_turbo >/dev/null 2>&1
if [ ! -d $ROOT/SpecForge/.git ]; then
  git clone --filter=blob:none https://github.com/sgl-project/SpecForge.git $ROOT/SpecForge 2>&1 | tail -1
fi
unset http_proxy https_proxy
cd $ROOT/SpecForge
git checkout 961ca7c3bb9d1d0a7aaf44db35e76e1f227a7be0 2>&1 | tail -1
python -m pip install --no-cache-dir -e . -i https://mirrors.bfsu.edu.cn/pypi/web/simple \
  --trusted-host mirrors.bfsu.edu.cn 2>&1 | tail -3
python -c "import transformers,sglang;print('tf',transformers.__version__,'sglang',sglang.__version__)"

echo "[$(date +%T)] === 6. Qwen3-1.7B ==="
export HF_HOME=$ROOT/cache/huggingface HF_ENDPOINT=https://hf-mirror.com
python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen3-1.7B",
    local_dir="/root/autodl-tmp/eagle3-qwen/models/Qwen3-1.7B",
    allow_patterns=["*.json","*.safetensors","*.txt","*.jinja","merges.txt","vocab.json"])
print("model at", p)
PY
du -sh $ROOT/models/Qwen3-1.7B
echo "[$(date +%T)] SETUP_DONE"
