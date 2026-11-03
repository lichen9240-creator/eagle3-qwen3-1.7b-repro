set -u
echo "=== OS ==="
. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME"
uname -r
echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,driver_version,power.limit --format=csv,noheader
echo "=== 磁盘 ==="
df -h /root/autodl-tmp /root 2>/dev/null | sed -n '1,4p'
echo "=== 计算资源 ==="
echo "cores=$(nproc)"
free -g | awk 'NR==2{print "mem_total_GB="$2}'
cat /sys/fs/cgroup/memory.max 2>/dev/null | awk '{printf "cgroup_mem_GB=%.0f\n",$1/1024/1024/1024}'
echo "=== 工具链 ==="
for t in git curl wget gcc g++ make cmake conda mamba uv pip3 python3 python; do
  p=$(command -v $t 2>/dev/null || true); [ -n "$p" ] && echo "$t -> $p"
done
echo "python3: $(python3 -V 2>&1)"
echo "=== conda envs ==="
conda env list 2>/dev/null | head -5 || echo "no conda"
echo "=== 网络 ==="
timeout 12 curl -sS -o /dev/null -w "pypi:%{http_code} %{time_total}s\n" https://pypi.org/simple/ 2>&1 || echo "pypi FAIL"
timeout 12 curl -sS -o /dev/null -w "github:%{http_code} %{time_total}s\n" https://github.com 2>&1 || echo "github FAIL"
timeout 12 curl -sS -o /dev/null -w "hf:%{http_code} %{time_total}s\n" https://huggingface.co 2>&1 || echo "hf FAIL"
timeout 12 curl -sS -o /dev/null -w "hf-mirror:%{http_code} %{time_total}s\n" https://hf-mirror.com 2>&1 || echo "hf-mirror FAIL"
echo "=== network_turbo ==="
ls -l /etc/network_turbo 2>/dev/null || echo "no /etc/network_turbo"
echo "=== 已有目录 ==="
ls -la /root/autodl-tmp 2>/dev/null | head -10
echo "=== ssh 端口/时间 ==="
date; echo "uptime: $(uptime -p 2>/dev/null)"
