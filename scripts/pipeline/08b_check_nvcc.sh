SP=/root/autodl-tmp/eagle3-qwen/envs/eagle3/lib/python3.12/site-packages
echo "=== cuda_nvcc 内容 ==="
ls -la $SP/nvidia/cuda_nvcc/ 2>/dev/null
echo "=== bin ==="
ls -la $SP/nvidia/cuda_nvcc/bin/ 2>/dev/null
echo "=== 全盘找 nvcc ==="
find $SP/nvidia -name 'nvcc*' 2>/dev/null | head -10
echo "=== pip 包版本 ==="
/root/autodl-tmp/eagle3-qwen/envs/eagle3/bin/python -m pip show nvidia-cuda-nvcc-cu12 2>/dev/null | head -4
echo "=== conda nvidia 源可达性 ==="
timeout 15 curl -s -o /dev/null -w "tuna-nvidia:%{http_code}\n" https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/nvidia/linux-64/repodata.json
