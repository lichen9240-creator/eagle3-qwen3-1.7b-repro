export PATH=/root/miniconda3/bin:$PATH
export PATH=/root/miniconda3/condabin:$PATH
echo "=== conda ==="
conda --version 2>&1 | head -2
echo "python(on path): $(command -v python) $(python -V 2>&1)"
echo "=== pip 源配置 ==="
cat /etc/pip.conf 2>/dev/null; cat ~/.pip/pip.conf 2>/dev/null; cat ~/.config/pip/pip.conf 2>/dev/null; echo "---"
echo "=== network_turbo 内容 ==="
cat /etc/network_turbo
echo "=== 开 turbo 后网络 ==="
source /etc/network_turbo
timeout 15 curl -sS -o /dev/null -w "pypi:%{http_code} %{time_total}s\n" https://pypi.org/simple/ 2>&1
timeout 15 curl -sS -o /dev/null -w "github:%{http_code} %{time_total}s\n" https://github.com 2>&1
timeout 15 curl -sS -o /dev/null -w "hf:%{http_code} %{time_total}s\n" https://huggingface.co 2>&1
echo "=== 国内镜像 ==="
timeout 15 curl -sS -o /dev/null -w "aliyun-pypi:%{http_code} %{time_total}s\n" https://mirrors.aliyun.com/pypi/simple/ 2>&1
timeout 15 curl -sS -o /dev/null -w "tuna-pypi:%{http_code} %{time_total}s\n" https://pypi.tuna.tsinghua.edu.cn/simple/ 2>&1
timeout 15 curl -sS -o /dev/null -w "download.pytorch:%{http_code} %{time_total}s\n" https://download.pytorch.org/whl/cu128 2>&1
echo "=== conda python 版本 ==="
/root/miniconda3/bin/python -V
