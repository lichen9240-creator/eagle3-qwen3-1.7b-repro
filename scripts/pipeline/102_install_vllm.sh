set -e
ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=/root/miniconda3/bin:$PATH
echo "[$(date +%T)] 建独立 vLLM 环境（不污染 sglang env）"
if [ ! -x $ROOT/envs/vllm/bin/python ]; then
  conda create -y -p $ROOT/envs/vllm python=3.12 2>&1 | tail -2
fi
export PATH=$ROOT/envs/vllm/bin:$PATH
python -V
echo "[$(date +%T)] 装 vllm==0.11.2"
python -m pip install -q -U pip
python -m pip install --no-cache-dir vllm==0.11.2 \
  -i https://mirrors.bfsu.edu.cn/pypi/web/simple --trusted-host mirrors.bfsu.edu.cn 2>&1 | tail -5
echo "[$(date +%T)] 版本"
python -c "import torch, vllm; print('torch', torch.__version__, torch.version.cuda); print('vllm', vllm.__version__)"
echo "[$(date +%T)] VLLM_ENV_DONE"
