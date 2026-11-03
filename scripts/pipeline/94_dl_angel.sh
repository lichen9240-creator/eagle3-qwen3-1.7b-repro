set -e
ROOT=/root/autodl-tmp/eagle3-qwen
pkill -9 -f 'sglang.launch_server' 2>/dev/null || true
sleep 5
export PATH=$ROOT/envs/eagle3/bin:$PATH
export HF_HOME=$ROOT/cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com
echo "[$(date +%T)] 下载 AngelSlim/Qwen3-1.7B_eagle3"
mkdir -p $ROOT/exports/angelslim_eagle3
python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("AngelSlim/Qwen3-1.7B_eagle3",
    local_dir="/root/autodl-tmp/eagle3-qwen/exports/angelslim_eagle3",
    allow_patterns=["*.json", "*.safetensors", "*.bin", "*.txt"])
print("downloaded to", p)
PY
echo "=== 文件 ==="
ls -lh $ROOT/exports/angelslim_eagle3/
echo "=== config.json ==="
cat $ROOT/exports/angelslim_eagle3/config.json 2>/dev/null
echo "[$(date +%T)] ANGEL_DOWNLOAD_DONE"
