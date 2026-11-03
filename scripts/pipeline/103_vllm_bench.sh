set -e
ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/vllm/bin:$PATH
export HF_HOME=$ROOT/cache/huggingface
mkdir -p $ROOT/runs/vllm
cd $ROOT

start_server() {
  MODE=$1
  pkill -9 -f 'vllm.entrypoints' 2>/dev/null || true
  pkill -9 -f 'sglang.launch_server' 2>/dev/null || true
  for i in $(seq 1 60); do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    [ "${m:-99999}" -lt 800 ] && break
    sleep 2
  done
  if [ "$MODE" = "base" ]; then
    nohup python -m vllm.entrypoints.openai.api_server \
      --model "$ROOT/models/Qwen3-1.7B" --served-model-name qwen3 \
      --dtype bfloat16 --gpu-memory-utilization 0.55 --max-model-len 4096 \
      --port 8000 > "$ROOT/logs/vllm_base.log" 2>&1 &
  else
    nohup python -m vllm.entrypoints.openai.api_server \
      --model "$ROOT/models/Qwen3-1.7B" --served-model-name qwen3 \
      --speculative-config "{\"model\": \"$ROOT/exports/angelslim_eagle3\", \"num_speculative_tokens\": 2}" \
      --dtype bfloat16 --gpu-memory-utilization 0.55 --max-model-len 4096 \
      --port 8000 > "$ROOT/logs/vllm_eagle.log" 2>&1 &
  fi
  for i in $(seq 1 120); do
    c=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8000/health || true)
    [ "$c" = "200" ] && { echo "health ok ($MODE)"; return 0; }
    sleep 5
  done
  echo "!! health failed ($MODE)"; tail -20 "$ROOT/logs/vllm_$MODE.log" | sed 's/[^[:print:]]//g' | cut -c1-180
  return 1
}

echo "[$(date +%T)] === vLLM baseline ==="
start_server base
python $ROOT/vllm_bench.py --base http://127.0.0.1:8000 --data $ROOT/data/bench/bench_90.jsonl \
  --model qwen3 --max-tokens 128 --requests 120 --conc 1 4 --out-dir $ROOT/runs/vllm --tag base
python $ROOT/vllm_bench.py --base http://127.0.0.1:8000 --data $ROOT/data/bench/bench_90.jsonl \
  --model qwen3 --max-tokens 1024 --requests 24 --conc 1 --out-dir $ROOT/runs/vllm --tag base1024

echo "[$(date +%T)] === vLLM EAGLE3 (AngelSlim, n=2) ==="
start_server eagle
grep -aE 'speculative|EAGLE' "$ROOT/logs/vllm_eagle.log" | tail -3 | cut -c1-160 || true
python $ROOT/vllm_bench.py --base http://127.0.0.1:8000 --data $ROOT/data/bench/bench_90.jsonl \
  --model qwen3 --max-tokens 128 --requests 120 --conc 1 4 --out-dir $ROOT/runs/vllm --tag eagle
python $ROOT/vllm_bench.py --base http://127.0.0.1:8000 --data $ROOT/data/bench/bench_90.jsonl \
  --model qwen3 --max-tokens 1024 --requests 24 --conc 1 --out-dir $ROOT/runs/vllm --tag eagle1024

pkill -9 -f 'vllm.entrypoints' 2>/dev/null || true
echo "[$(date +%T)] VLLM_BENCH_DONE"
