set -e
ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/vllm/bin:$PATH
export HF_HOME=$ROOT/cache/huggingface
cd $ROOT

echo "[$(date +%T)] 清理所有 vllm/sglang 进程"
pkill -9 -f 'vllm' 2>/dev/null || true
pkill -9 -f 'sglang' 2>/dev/null || true
for i in $(seq 1 90); do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  echo "  used=${m} MiB"
  [ "${m:-99999}" -lt 800 ] && break
  sleep 2
done

echo "[$(date +%T)] 启动 vLLM + EAGLE3 (AngelSlim, num_speculative_tokens=2)"
nohup python -m vllm.entrypoints.openai.api_server \
  --model "$ROOT/models/Qwen3-1.7B" --served-model-name qwen3 \
  --speculative-config "{\"model\": \"$ROOT/exports/angelslim_eagle3\", \"num_speculative_tokens\": 2}" \
  --dtype bfloat16 --gpu-memory-utilization 0.50 --max-model-len 4096 \
  --port 8000 > "$ROOT/logs/vllm_eagle2.log" 2>&1 &
ok=0
for i in $(seq 1 120); do
  c=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8000/health || true)
  [ "$c" = "200" ] && { ok=1; break; }
  sleep 5
done
if [ "$ok" != "1" ]; then
  echo "!! health failed"
  grep -aiE 'ValueError|RuntimeError|error' "$ROOT/logs/vllm_eagle2.log" | tail -5 | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-200
  exit 1
fi
echo "health ok"
grep -aiE 'speculative|eagle' "$ROOT/logs/vllm_eagle2.log" | tail -3 | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-160

python $ROOT/vllm_bench.py --base http://127.0.0.1:8000 --data $ROOT/data/bench/bench_90.jsonl \
  --model qwen3 --max-tokens 128 --requests 120 --conc 1 4 --out-dir $ROOT/runs/vllm --tag eagle
python $ROOT/vllm_bench.py --base http://127.0.0.1:8000 --data $ROOT/data/bench/bench_90.jsonl \
  --model qwen3 --max-tokens 1024 --requests 24 --conc 1 --out-dir $ROOT/runs/vllm --tag eagle1024

pkill -9 -f 'vllm' 2>/dev/null || true
echo "[$(date +%T)] VLLM_EAGLE_DONE"
