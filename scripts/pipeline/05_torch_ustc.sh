ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
pkill -f 'pip install' || true
sleep 3
echo "[$(date +%T)] 改用 ustc 源重装 torch 栈"
python -m pip install --no-cache-dir --retries 10 --timeout 60 \
  torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
  -i https://mirrors.ustc.edu.cn/pypi/simple --trusted-host mirrors.ustc.edu.cn
echo "[$(date +%T)] torch 验证"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory // 2**30, "GB")
PY
echo "[$(date +%T)] TORCH_DONE"
