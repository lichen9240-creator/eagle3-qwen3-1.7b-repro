ROOT=/root/autodl-tmp/eagle3-qwen
PY=$ROOT/envs/eagle3/bin/python
cd /tmp
for v in 12.4.127 12.6.77; do
  echo "=== 试 nvidia-cuda-nvcc-cu12==$v ==="
  rm -rf /tmp/nv_$v; mkdir -p /tmp/nv_$v
  timeout 180 $PY -m pip download --no-deps -q -d /tmp/nv_$v \
    nvidia-cuda-nvcc-cu12==$v -i https://mirrors.ustc.edu.cn/pypi/simple \
    --trusted-host mirrors.ustc.edu.cn 2>&1 | tail -2
  f=$(ls /tmp/nv_$v/*.whl 2>/dev/null | head -1)
  if [ -n "$f" ]; then
    echo "wheel: $(basename $f) $(du -h $f | cut -f1)"
    $PY - "$f" <<'PY'
import sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
names = z.namelist()
nv = [n for n in names if n.endswith('/bin/nvcc') or n.endswith('/bin/ptxas')]
inc = [n for n in names if 'include' in n and n.endswith('cuda_runtime.h')]
print("  nvcc/ptxas:", nv[:3])
print("  cuda_runtime.h in wheel:", inc[:2])
print("  total files:", len(names))
PY
  else
    echo "  下载失败"
  fi
done
