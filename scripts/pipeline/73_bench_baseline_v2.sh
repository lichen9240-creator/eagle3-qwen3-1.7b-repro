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
sleep 6
echo "[$(date +%T)] baseline 服务"
nohup python -m sglang.launch_server \
  --model-path "$ROOT/models/Qwen3-1.7B" \
  --dtype bfloat16 --mem-fraction-static 0.55 \
  --max-running-requests 8 --disable-radix-cache \
  --host 127.0.0.1 --port 30000 > "$ROOT/logs/baseline_v2_server.log" 2>&1 &
echo $! > "$ROOT/logs/baseline_v2_server.pid"
for i in $(seq 1 120); do
  c=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:30000/health || true)
  [ "$c" = "200" ] && { echo "[$(date +%T)] HEALTH_200"; break; }
  sleep 5
done
mkdir -p $ROOT/runs/baseline_v2
echo "[$(date +%T)] baseline 正确性"
python "$ROOT/repro/scripts/check_outputs.py" \
  --data "$ROOT/data/bench/bench_90.jsonl" \
  --out "$ROOT/runs/baseline_v2_correctness.json" --n 10 --max-tokens 64
echo "[$(date +%T)] baseline 吞吐"
python "$ROOT/repro/scripts/bench_serving.py" \
  --url http://127.0.0.1:30000/generate \
  --data "$ROOT/data/bench/bench_90.jsonl" \
  --out-dir "$ROOT/runs/baseline_v2" --tag baseline \
  --concurrency 1 4 8 --requests 90 --max-tokens 128 --warmup 3
echo "[$(date +%T)] BENCH_BASELINE_V2_DONE"
