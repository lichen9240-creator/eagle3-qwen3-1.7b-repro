set -e
ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=/root/miniconda3/bin:$PATH
echo "[$(date +%T)] conda 安装 cuda-nvcc 12.8"
conda create -y -p $ROOT/envs/cuda128 -c nvidia \
  cuda-nvcc=12.8 cuda-cudart-dev=12.8 cuda-cccl=12.8 2>&1 | tail -25
echo "[$(date +%T)] 检查"
ls $ROOT/envs/cuda128/bin/ | grep -E '^nvcc|ptxas' || true
$ROOT/envs/cuda128/bin/nvcc --version | tail -2
