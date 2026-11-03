set -x
echo "=== 1. 测到目标实例的 TCP 连通性 ==="
timeout 10 bash -c 'cat < /dev/null > /dev/tcp/connect.westd.seetacloud.com/11570' && echo TCP_OK || echo TCP_FAIL

echo "=== 2. sshpass 是否已装 ==="
command -v sshpass || apt-get install -y -qq sshpass 2>&1 | tail -3
command -v sshpass

echo "=== 3. 测带宽（传 1MB 计时）==="
time dd if=/dev/zero of=/tmp/_bwtest.bin bs=1M count=20 2>/dev/null
time sshpass -p "${TARGET_SSH_PW:?export TARGET_SSH_PW}" scp -P 11570 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  /tmp/_bwtest.bin root@connect.westd.seetacloud.com:/tmp/_bwtest.bin 2>&1 | tail -3
rm -f /tmp/_bwtest.bin
