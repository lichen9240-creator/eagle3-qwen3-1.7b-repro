ROOT=/root/autodl-tmp/eagle3-qwen
PY=$ROOT/envs/eagle3/bin/python
declare -A M=(
  [aliyun]="http://mirrors.aliyun.com/pypi/simple"
  [tuna]="https://pypi.tuna.tsinghua.edu.cn/simple"
  [ustc]="https://mirrors.ustc.edu.cn/pypi/simple"
  [huawei]="https://repo.huaweicloud.com/repository/pypi/simple"
  [tencent]="https://mirrors.cloud.tencent.com/pypi/simple"
)
for k in aliyun tuna ustc huawei tencent; do
  url=${M[$k]}
  rm -rf /tmp/mt_$k; mkdir -p /tmp/mt_$k
  s=$(date +%s)
  timeout 45 $PY -m pip download transformers --no-deps -d /tmp/mt_$k \
      -i "$url" --trusted-host "$(echo $url | sed -E 's#https?://([^/]+)/.*#\1#')" -q >/dev/null 2>&1
  rc=$?
  e=$(date +%s); d=$((e-s))
  sz=$(du -sm /tmp/mt_$k 2>/dev/null | cut -f1)
  printf "%-8s rc=%s %3ss %4sMB -> %s\n" "$k" "$rc" "$d" "${sz:-0}" "$([ "${sz:-0}" -gt 0 ] && echo "$((sz*1024/(d>0?d:1)))KB/s" || echo 0KB/s)"
done
