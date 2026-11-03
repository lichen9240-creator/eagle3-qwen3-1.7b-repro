set -e
ROOT=/root/autodl-tmp/eagle3-qwen
OUT=$ROOT/data/ultrachat_full/ultrachat_30k_regen.jsonl
TARGET=15000
echo "[$(date +%T)] 守候：生成到 $TARGET 条后自动停止"
while true; do
  n=$(wc -l < "$OUT" 2>/dev/null || echo 0)
  if [ "$n" -ge "$TARGET" ]; then
    echo "[$(date +%T)] 已达 $n 条，停止生成"
    pkill -f 'regenerate_train_data.py' || true
    sleep 3
    echo "[$(date +%T)] STOPPED_AT=$(wc -l < $OUT)"
    break
  fi
  sleep 15
done
