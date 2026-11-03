set -x
pkill -9 -f 'sglang.launch_server' || true
sleep 10
pgrep -fa sglang | head -3 || echo "no sglang"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
rm -f /root/autodl-tmp/eagle3-qwen/logs/regen_server.log
