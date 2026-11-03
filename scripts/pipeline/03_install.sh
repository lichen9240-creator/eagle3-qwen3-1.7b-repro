set -e
ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
export HF_HOME=$ROOT/cache/huggingface
cd $ROOT/SpecForge
echo "[$(date +%T)] pip upgrade"
python -m pip install -q -U pip setuptools wheel \
  -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
echo "[$(date +%T)] install SpecForge (torch 2.9.1 等，较大)"
python -m pip install -e . -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
echo "[$(date +%T)] verify"
python - <<'PY'
import torch, transformers, sglang
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0))
print("transformers", transformers.__version__, "sglang", sglang.__version__)
PY
echo "[$(date +%T)] INSTALL_DONE"
