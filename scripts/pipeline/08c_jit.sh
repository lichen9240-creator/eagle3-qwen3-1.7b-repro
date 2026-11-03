SP=/root/autodl-tmp/eagle3-qwen/envs/eagle3/lib/python3.12/site-packages
echo "=== 谁在调用 JIT ==="
grep -rn "sgl_kernel_jit\|jit_kernel" $SP/sglang/srt/ --include=*.py 2>/dev/null | head -12
echo
echo "=== 该 kernel 是否有原生分支 ==="
grep -rn "apply_rope_pos_ids_cos_sin_cache" $SP/sglang/srt/ --include=*.py 2>/dev/null | head -12
echo
echo "=== 相关环境变量开关 ==="
grep -rn "SGLANG_.*JIT\|DISABLE_JIT\|use_jit" $SP/sglang/srt/ $SP/sglang/ --include=*.py 2>/dev/null | head -12
echo
echo "=== 源可达性 ==="
for u in https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/ \
         https://conda.anaconda.org/nvidia/linux-64/repodata.json \
         https://mirrors.bfsu.edu.cn/anaconda/cloud/nvidia/linux-64/repodata.json \
         https://mirrors.aliyun.com/anaconda/cloud/nvidia/linux-64/repodata.json ; do
  timeout 12 curl -s -o /dev/null -w "%{http_code}  $u\n" "$u"
done
