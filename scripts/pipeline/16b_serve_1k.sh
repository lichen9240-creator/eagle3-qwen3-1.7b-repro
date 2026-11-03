set -e
ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
export CUDA_HOME=$ROOT/envs/cuda128
export CUDACXX=$CUDA_HOME/bin/nvcc
export PATH=$ROOT/envs/hostbin:$CUDA_HOME/bin:$PATH
export NVCC_PREPEND_FLAGS="-ccbin $ROOT/envs/hostbin/g++"
[ -d "$CUDA_HOME/lib" ] && ln -sfn "$CUDA_HOME/lib" "$CUDA_HOME/lib64"
export HF_HOME=$ROOT/cache/huggingface
cd $ROOT

CKPT=$(ls -d "$ROOT/runs/smoke_1k"/epoch_* 2>/dev/null | sort -V | tail -1)
[ -n "$CKPT" ] || { echo "NO_CKPT"; exit 1; }
echo "CKPT=$CKPT"
ls -lh "$CKPT"

if [ -f "$ROOT/logs/smoke_1k_server.pid" ]; then kill $(cat "$ROOT/logs/smoke_1k_server.pid") 2>/dev/null || true; sleep 5; fi

nohup python -m sglang.launch_server \
  --model-path "$ROOT/models/Qwen3-1.7B" \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path "$CKPT" \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --dtype bfloat16 --mem-fraction-static 0.55 \
  --max-running-requests 8 --disable-radix-cache \
  --host 127.0.0.1 --port 30000 \
  > "$ROOT/logs/smoke_1k_server.log" 2>&1 &
echo $! > "$ROOT/logs/smoke_1k_server.pid"
echo "[$(date +%T)] pid=$(cat $ROOT/logs/smoke_1k_server.pid)"

for i in $(seq 1 120); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:30000/health || true)
  if [ "$code" = "200" ]; then echo "[$(date +%T)] HEALTH_200"; exit 0; fi
  sleep 5
done
echo "HEALTH_TIMEOUT"; tail -40 "$ROOT/logs/smoke_1k_server.log"; exit 1
