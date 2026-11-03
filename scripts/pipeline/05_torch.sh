set -e
ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
echo "[$(date +%T)] 停止慢速 pip"
pkill -f 'pip install -e' || true
sleep 3
rm -rf /tmp/pip-unpack-* /tmp/pip-install-* 2>/dev/null || true
echo "[$(date +%T)] 从 download.pytorch.org 装 torch 栈"
python -m pip install --no-cache-dir \
  torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu128
echo "[$(date +%T)] torch 验证"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0))
PY
echo "[$(date +%T)] TORCH_DONE"
