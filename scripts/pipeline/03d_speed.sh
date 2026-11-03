echo "=== 当前 torch 下载进度 ==="
ls -la /tmp/pip-unpack-*/ 2>/dev/null | grep -E 'torch|total' | head -3
du -sm /tmp/pip-unpack-* 2>/dev/null

echo "=== 各源单连接速度（8s 采样）==="
for url in \
  "https://download.pytorch.org/whl/cu128/torch-2.9.1%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl" \
  "https://mirrors.ustc.edu.cn/pypi/web/simple/torch/" \
  "https://mirrors.cernet.edu.cn/pypi/web/simple/torch/" ; do
  echo -n "$url -> "
  timeout 12 curl -s -o /dev/null -w "http=%{http_code} speed=%{speed_download}B/s\n" --max-time 8 -r 0-52428800 "$url" 2>&1
done

echo "=== 阿里云并发 4 连接合计 ==="
for i in 0 1 2 3; do
  ( timeout 10 curl -s -o /dev/null -w "conn$i=%{speed_download}\n" --max-time 8 \
      -r $((i*4194304))-$((i*4194304+4194303)) \
      "http://mirrors.aliyun.com/pypi/packages/source/t/torch/" 2>&1 ) &
done
wait
echo "=== hf 速度复核 ==="
timeout 12 curl -s -o /dev/null -w "hf-mirror speed=%{speed_download}B/s\n" --max-time 8 -r 0-52428800 \
  "https://hf-mirror.com/Qwen/Qwen3-1.7B/resolve/main/model-00002-of-00002.safetensors" 2>&1
