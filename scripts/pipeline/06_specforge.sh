set -e
ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
export HF_HOME=$ROOT/cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com
cd $ROOT/SpecForge
echo "[$(date +%T)] 用 ustc 源安装 SpecForge 依赖（torch 已就绪）"
python -m pip install --no-cache-dir -e . \
  -i https://mirrors.ustc.edu.cn/pypi/simple --trusted-host mirrors.ustc.edu.cn
echo "[$(date +%T)] ==== G0.4 版本验收 ===="
python - <<'PY'
import torch, transformers, sglang, sys
print("python", sys.version.split()[0])
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory // 2**30, "GB")
print("transformers", transformers.__version__)
print("sglang", sglang.__version__)
ok = (torch.__version__.startswith("2.9.1") and torch.version.cuda == "12.8"
      and torch.cuda.is_available() and transformers.__version__ == "4.57.1"
      and sglang.__version__ == "0.5.9")
print("G0_VERSION_OK" if ok else "G0_VERSION_MISMATCH")
PY
echo "[$(date +%T)] SPECFORGE_DONE"
