set -e
ROOT=/root/autodl-tmp/eagle3-qwen
export HF_HOME=$ROOT/cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com
export PATH=/root/miniconda3/bin:$PATH
echo "[$(date +%T)] install huggingface_hub into base py3.8"
python -m pip install -q -U "huggingface_hub[hf_transfer]" \
  -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
echo "[$(date +%T)] downloading Qwen/Qwen3-1.7B from hf-mirror"
python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download(
    repo_id="Qwen/Qwen3-1.7B",
    local_dir="/root/autodl-tmp/eagle3-qwen/models/Qwen3-1.7B",
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "merges.txt", "vocab.json"],
    max_workers=8,
)
print("downloaded to", p)
PY
echo "[$(date +%T)] MODEL_DONE"
du -sh $ROOT/models/Qwen3-1.7B
ls -lh $ROOT/models/Qwen3-1.7B | head -20
