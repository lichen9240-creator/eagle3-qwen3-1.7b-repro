echo "=== pip 临时文件 ==="
ls -la /tmp 2>/dev/null | grep -i pip | head -5
echo "=== cache 占用（间隔 12s 采样）==="
for i in 1 2 3; do
  du -sm /root/.cache/pip 2>/dev/null | awk -v i=$i '{print "sample"i": "$1" MB"}'
  sleep 6
done
echo "=== 下载中的临时 html/whl ==="
find /tmp -name '*.whl*' -o -name 'pip-*' 2>/dev/null | head -5
echo "=== 进程 ==="
ps -eo pid,etime,pcpu,cmd | grep -E 'pip (install|-m)' | grep -v grep | head -3
echo "=== 到阿里云的实测速度(100MB / 6s) ==="
timeout 20 curl -o /dev/null -s -w "speed=%{speed_download} B/s http=%{http_code}\n" \
  --max-time 15 -r 0-104857600 http://mirrors.aliyun.com/pypi/packages/ 2>&1
echo "=== 到 tuna 的实测速度 ==="
timeout 20 curl -o /dev/null -s -w "speed=%{speed_download} B/s http=%{http_code}\n" --max-time 15 \
  https://pypi.tuna.tsinghua.edu.cn/simple/torch/ 2>&1
echo "=== 模型文件 ==="
ls -lh /root/autodl-tmp/eagle3-qwen/models/Qwen3-1.7B
