set -e
echo "=== apt 源 ==="
grep -h '^deb ' /etc/apt/sources.list 2>/dev/null | head -3
echo "=== 是否已有 gcc-10 ==="
ls /usr/bin/gcc-1* 2>/dev/null || echo "none"
echo "=== apt update (30s) ==="
timeout 60 apt-get update -qq 2>&1 | tail -3 || echo "APT_UPDATE_FAIL"
echo "=== 安装 g++-10 ==="
DEBIAN_FRONTEND=noninteractive timeout 300 apt-get install -y -qq gcc-10 g++-10 2>&1 | tail -5
echo "=== 验证 ==="
/usr/bin/g++-10 --version | head -1
printf '#include <concepts>\n#include <cstdio>\ntemplate<class T> concept I = std::integral<T>;\nint main(){puts("ok");}\n' > /tmp/c20.cpp
/usr/bin/g++-10 -std=c++20 /tmp/c20.cpp -o /tmp/c20 && /tmp/c20
