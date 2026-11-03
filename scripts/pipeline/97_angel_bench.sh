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
mkdir -p $ROOT/runs/angel

teardown() {
  pkill -9 -f 'sglang.launch_server' 2>/dev/null || true
  for i in $(seq 1 60); do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    [ "${m:-99999}" -lt 800 ] && return 0
    sleep 2
  done
}

run_case() {
  TAG=$1; STEPS=$2; TOPK=$3; DTOK=$4
  echo "[$(date +%T)] ===== $TAG (steps=$STEPS topk=$TOPK tokens=$DTOK) ====="
  teardown
  nohup python -m sglang.launch_server \
    --model-path "$ROOT/models/Qwen3-1.7B" \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path "$ROOT/exports/angelslim_eagle3" \
    --speculative-num-steps $STEPS --speculative-eagle-topk $TOPK \
    --speculative-num-draft-tokens $DTOK \
    --dtype bfloat16 --mem-fraction-static 0.55 --max-running-requests 8 \
    --disable-radix-cache --host 127.0.0.1 --port 30000 \
    > "$ROOT/logs/angel_${TAG}.log" 2>&1 &
  ok=0
  for i in $(seq 1 90); do
    c=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:30000/health || true)
    [ "$c" = "200" ] && { ok=1; break; }
    sleep 4
  done
  if [ "$ok" != "1" ]; then echo "!! health failed"; tail -8 "$ROOT/logs/angel_${TAG}.log" | sed 's/[^[:print:]]//g' | cut -c1-180; return 0; fi
  echo "   health ok"
  sleep 10
  python "$ROOT/repro/scripts/bench_serving.py" \
    --url http://127.0.0.1:30000/generate --data "$ROOT/data/bench/bench_90.jsonl" \
    --out-dir "$ROOT/runs/angel/$TAG" --tag "$TAG" \
    --concurrency 1 4 --requests 120 --max-tokens 128 --warmup 5
}

run_case angel_k2 2 1 3
run_case angel_k3 3 1 4
run_case angel_t2_d8 3 2 8
teardown
echo "[$(date +%T)] ANGELBENCH_DONE"
