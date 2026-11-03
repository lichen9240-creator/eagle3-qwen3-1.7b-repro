set -x
ls -l /usr/bin/gcc-10 /usr/bin/g++-10 2>&1 || echo "gcc-10 missing"
apt-get update -qq 2>&1 | tail -2
DEBIAN_FRONTEND=noninteractive apt-get install -y gcc-10 g++-10 2>&1 | tail -4
/usr/bin/g++-10 --version | head -1
ROOT=/root/autodl-tmp/eagle3-qwen
mkdir -p $ROOT/envs/hostbin
ln -sf /usr/bin/gcc-10 $ROOT/envs/hostbin/gcc
ln -sf /usr/bin/g++-10 $ROOT/envs/hostbin/g++
ln -sf /usr/bin/gcc-10 $ROOT/envs/hostbin/cc
ln -sf /usr/bin/g++-10 $ROOT/envs/hostbin/c++
ls -l $ROOT/envs/hostbin/
printf '#include <concepts>\nint main(){}\n' > /tmp/c20.cpp
$ROOT/envs/hostbin/g++ -std=c++20 /tmp/c20.cpp -o /tmp/c20 && echo "CXX20_OK"
