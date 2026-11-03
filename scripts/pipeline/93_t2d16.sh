set -e
ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
export CUDA_HOME=$ROOT/envs/cuda128
export CUDACXX=$CUDA_HOME/bin/nvcc
export PATH=$ROOT/envs/hostbin:$CUDA_HOME/bin:$PATH
export NVCC_PREPEND_FLAGS="-ccbin $ROOT/envs/hostbin/g++"
[ -d "$CUDA_HOME/lib" ] && ln -sfn "$CUDA_HOME/lib" "$CUDA_HOME/lib64"
export HF_HOME=$ROOT/cache/huggingface
cd $ROOT/SpecForge
pkill -9 -f 'sglang.launch_server' 2>/dev/null || true
for i in $(seq 1 60); do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [ "${m:-99999}" -lt 800 ] && break
  sleep 2
done
echo "[$(date +%T)] v2_t2_d16 (mem-fraction 0.45)"
nohup python -m sglang.launch_server \
  --model-path "$ROOT/models/Qwen3-1.7B" \
  --speculative-algorithm EAGLE3 --speculative-draft-model-path "$ROOT/exports/qwen3-1.7b-eagle3-v2-42k" \
  --speculative-num-steps 3 --speculative-eagle-topk 2 --speculative-num-draft-tokens 16 \
  --dtype bfloat16 --mem-fraction-static 0.45 --max-running-requests 8 \
  --disable-radix-cache --host 127.0.0.1 --port 30000 \
  > "$ROOT/logs/tree_t2d16.log" 2>&1 &
for i in $(seq 1 90); do
  c=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:30000/health || true)
  [ "$c" = "200" ] && { echo "health ok"; break; }
  sleep 4
done
sleep 10
python "$ROOT/repro/scripts/bench_serving.py" \
  --url http://127.0.0.1:30000/generate --data "$ROOT/data/bench/bench_90.jsonl" \
  --out-dir "$ROOT/runs/treesweep/v2_t2_d16" --tag v2_t2_d16 \
  --concurrency 1 4 --requests 120 --max-tokens 128 --warmup 5
pkill -9 -f 'sglang.launch_server' 2>/dev/null || true
echo "[$(date +%T)] T2D16_DONE"
