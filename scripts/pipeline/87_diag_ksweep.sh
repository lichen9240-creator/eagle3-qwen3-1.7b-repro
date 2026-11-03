ROOT=/root/autodl-tmp/eagle3-qwen
for t in baseline v1_k2 v1_k3 v1_k4 v2_k2 v2_k3 v2_k4; do
  f=$ROOT/logs/ksweep_$t.log
  echo "=== $t ==="
  grep -a 'speculative-num-steps\|speculative-num-draft-tokens' "$f" 2>/dev/null | head -1 | cut -c1-200
  grep -ac 'address already in use' "$f" 2>/dev/null
  grep -a 'Load weight end' "$f" 2>/dev/null | tail -1 | cut -c1-110
done
echo "=== 当前 GPU ==="
nvidia-smi --query-gpu=memory.used --format=csv,noheader
