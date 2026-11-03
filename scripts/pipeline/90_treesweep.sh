set -e
ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
export CUDA_HOME=$ROOT/envs/cuda128
export CUDACXX=$CUDA_HOME/bin/nvcc
export PATH=$ROOT/envs/hostbin:$CUDA_HOME/bin:$PATH
export NVCC_PREPEND_FLAGS="-ccbin $ROOT/envs/hostbin/g++"
[ -d "$CUDA_HOME/lib" ] && ln -sfn "$CUDA_HOME/lib" "$CUDA_HOME/lib64"
export HF_HOME=$ROOT/cache/huggingface
mkdir -p $ROOT/runs/treesweep
cd $ROOT/SpecForge

teardown() {
  pkill -9 -f 'sglang.launch_server' 2>/dev/null || true
  for i in $(seq 1 60); do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    if [ "${m:-99999}" -lt 800 ]; then return 0; fi
    sleep 2
  done
}

run_case() {
  TAG=$1; DRAFT=$2; STEPS=$3; TOPK=$4; DTOK=$5
  echo "[$(date +%T)] ===== $TAG (steps=$STEPS topk=$TOPK draft_tokens=$DTOK) ====="
  teardown
  nohup python -m sglang.launch_server \
    --model-path "$ROOT/models/Qwen3-1.7B" \
    --speculative-algorithm EAGLE3 --speculative-draft-model-path "$DRAFT" \
    --speculative-num-steps $STEPS --speculative-eagle-topk $TOPK \
    --speculative-num-draft-tokens $DTOK \
    --dtype bfloat16 --mem-fraction-static 0.55 --max-running-requests 8 \
    --disable-radix-cache --host 127.0.0.1 --port 30000 \
    > "$ROOT/logs/tree_${TAG}.log" 2>&1 &
  ok=0
  for i in $(seq 1 90); do
    c=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:30000/health || true)
    [ "$c" = "200" ] && { ok=1; break; }
    sleep 4
  done
  if [ "$ok" != "1" ]; then echo "!! health failed"; tail -5 "$ROOT/logs/tree_${TAG}.log" | cut -c1-160; return 0; fi
  echo -n "   effective: "
  ps -eo args | grep 'sglang.launch_server' | grep -v grep | head -1 | \
    grep -oE 'eagle-topk [0-9]+|num-steps [0-9]+|num-draft-tokens [0-9]+' | tr '\n' ' '
  echo
  sleep 10
  python "$ROOT/repro/scripts/bench_serving.py" \
    --url http://127.0.0.1:30000/generate --data "$ROOT/data/bench/bench_90.jsonl" \
    --out-dir "$ROOT/runs/treesweep/$TAG" --tag "$TAG" \
    --concurrency 1 4 --requests 120 --max-tokens 128 --warmup 5
}

# v2：topk=2 树（8 节点）、topk=4 树（16 节点）
run_case v2_t2_d8  "$ROOT/exports/qwen3-1.7b-eagle3-v2-42k" 3 2 8
run_case v2_t4_d16 "$ROOT/exports/qwen3-1.7b-eagle3-v2-42k" 3 4 16
run_case v2_t2_d16 "$ROOT/exports/qwen3-1.7b-eagle3-v2-42k" 3 2 16
teardown
echo "[$(date +%T)] TREESWEEP_DONE"
