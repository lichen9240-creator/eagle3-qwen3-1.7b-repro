set -e
ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
export CUDA_HOME=$ROOT/envs/cuda128
export CUDACXX=$CUDA_HOME/bin/nvcc
export PATH=$ROOT/envs/hostbin:$CUDA_HOME/bin:$PATH
export NVCC_PREPEND_FLAGS="-ccbin $ROOT/envs/hostbin/g++"
[ -d "$CUDA_HOME/lib" ] && ln -sfn "$CUDA_HOME/lib" "$CUDA_HOME/lib64"
export HF_HOME=$ROOT/cache/huggingface
mkdir -p $ROOT/runs/ksweep2
cd $ROOT/SpecForge

teardown() {
  pkill -9 -f 'sglang.launch_server' 2>/dev/null || true
  for i in $(seq 1 60); do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    if [ "${m:-99999}" -lt 800 ]; then return 0; fi
    sleep 2
  done
  echo "WARN: GPU 未释放 ($m MiB)"; return 0
}

run_case() {
  TAG=$1; DRAFT=$2; STEPS=$3; DTOK=$4
  echo "[$(date +%T)] ===== $TAG ====="
  teardown
  if [ -z "$DRAFT" ]; then
    nohup python -m sglang.launch_server \
      --model-path "$ROOT/models/Qwen3-1.7B" --dtype bfloat16 \
      --mem-fraction-static 0.55 --max-running-requests 8 --disable-radix-cache \
      --host 127.0.0.1 --port 30000 > "$ROOT/logs/ks2_${TAG}.log" 2>&1 &
  else
    nohup python -m sglang.launch_server \
      --model-path "$ROOT/models/Qwen3-1.7B" \
      --speculative-algorithm EAGLE3 --speculative-draft-model-path "$DRAFT" \
      --speculative-num-steps $STEPS --speculative-eagle-topk 1 \
      --speculative-num-draft-tokens $DTOK \
      --dtype bfloat16 --mem-fraction-static 0.55 --max-running-requests 8 \
      --disable-radix-cache --host 127.0.0.1 --port 30000 \
      > "$ROOT/logs/ks2_${TAG}.log" 2>&1 &
  fi
  ok=0
  for i in $(seq 1 90); do
    c=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:30000/health || true)
    [ "$c" = "200" ] && { ok=1; break; }
    sleep 4
  done
  if [ "$ok" != "1" ]; then echo "!! $TAG health failed"; tail -5 "$ROOT/logs/ks2_${TAG}.log" | cut -c1-160; return 0; fi
  # 校验实际生效的参数
  echo -n "   effective: "
  ps -eo args | grep 'sglang.launch_server' | grep -v grep | head -1 | \
    grep -oE 'speculative-num-steps [0-9]+|speculative-num-draft-tokens [0-9]+|speculative-draft-model-path \S+' | tr '\n' ' '
  echo
  if grep -aq 'address already in use' "$ROOT/logs/ks2_${TAG}.log"; then echo "  !! PORT CONFLICT"; fi
  sleep 10
  python "$ROOT/repro/scripts/bench_serving.py" \
    --url http://127.0.0.1:30000/generate --data "$ROOT/data/bench/bench_90.jsonl" \
    --out-dir "$ROOT/runs/ksweep2/$TAG" --tag "$TAG" \
    --concurrency 1 4 --requests 120 --max-tokens 128 --warmup 5
}

run_case baseline "" 0 0
run_case v1_k2 "$ROOT/exports/qwen3-1.7b-eagle3-sglang" 2 3
run_case v1_k3 "$ROOT/exports/qwen3-1.7b-eagle3-sglang" 3 4
run_case v1_k4 "$ROOT/exports/qwen3-1.7b-eagle3-sglang" 4 5
run_case v2_k2 "$ROOT/exports/qwen3-1.7b-eagle3-v2-42k" 2 3
run_case v2_k3 "$ROOT/exports/qwen3-1.7b-eagle3-v2-42k" 3 4
run_case v2_k4 "$ROOT/exports/qwen3-1.7b-eagle3-v2-42k" 4 5
teardown
echo "[$(date +%T)] KSWEEP2_DONE"
