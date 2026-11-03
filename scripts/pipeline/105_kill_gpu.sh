set -x
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' '); do
  echo "killing GPU pid $pid"
  kill -9 "$pid" 2>/dev/null || true
done
pkill -9 -f vllm 2>/dev/null || true
pkill -9 -f sglang 2>/dev/null || true
sleep 8
nvidia-smi --query-gpu=memory.used --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | head -3 || echo "no apps"
