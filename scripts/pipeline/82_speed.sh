echo "=== 各源单连接速度（8 秒采样，取 50MB 区间）==="
for u in \
  "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/torch/" \
  "https://pypi.tuna.tsinghua.edu.cn/simple/torch/" \
  "https://mirrors.aliyun.com/pypi/simple/torch/" \
  "https://mirrors.bfsu.edu.cn/pypi/web/simple/torch/" ; do
  echo -n "$u -> "
  timeout 12 curl -s -o /dev/null -w "%{http_code} %{speed_download}B/s\n" --max-time 8 -r 0-52428800 "$u" 2>&1
done
echo "=== 当前 pip 下载进度 ==="
du -sm /tmp/pip-* 2>/dev/null | sort -n | tail -2
