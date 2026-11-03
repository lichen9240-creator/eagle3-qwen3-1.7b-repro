ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
echo "[$(date +%T)] 安装 nvcc 12.8 (pip)"
python -m pip install -q --no-cache-dir nvidia-cuda-nvcc-cu12==12.8.93 nvidia-cuda-cccl-cu12==12.8.90 \
  -i https://mirrors.ustc.edu.cn/pypi/simple --trusted-host mirrors.ustc.edu.cn 2>&1 | tail -3
SP=$ROOT/envs/eagle3/lib/python3.12/site-packages
NVB=$SP/nvidia/cuda_nvcc/bin
echo "nvcc path: $NVB/nvcc"
ls $NVB | head -5
$NVB/nvcc --version | tail -2
echo "--- 目录结构 ---"
ls $SP/nvidia/cuda_nvcc/
echo "=== 测试1: c++20 ==="
mkdir -p /tmp/nvtest && cd /tmp/nvtest
printf '__global__ void k(){}\n' > t1.cu
$NVB/nvcc -std=c++20 -arch=sm_89 -c t1.cu -o t1.o && echo "CPP20_OK"
echo "=== 测试2: cuda_runtime.h + fp16 ==="
printf '#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n__global__ void k(){__half h;}\n' > t2.cu
$NVB/nvcc -std=c++20 -arch=sm_89 -c t2.cu -o t2.o && echo "HEADER_OK" || echo "HEADER_FAIL"
