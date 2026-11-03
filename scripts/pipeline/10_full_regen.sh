set -e
ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
export HF_HOME=$ROOT/cache/huggingface
cd $ROOT/SpecForge
mkdir -p $ROOT/data/regen

echo "[$(date +%T)] === G3.1 train 28.5k regeneration ==="
python scripts/regenerate_train_data.py \
  --model Qwen3-1.7B \
  --concurrency 32 \
  --max-tokens 512 \
  --temperature 0.8 \
  --top-p 0.95 \
  --top-k 20 \
  --server-address 127.0.0.1:30000 \
  --input-file-path "$ROOT/data/sharegpt_30k/sharegpt_30k_train.jsonl" \
  --output-file-path "$ROOT/data/regen/sharegpt_30k_train_regen.jsonl"
echo "[$(date +%T)] train done: $(wc -l < $ROOT/data/regen/sharegpt_30k_train_regen.jsonl) rows"

echo "[$(date +%T)] === G3.2 eval 1.5k regeneration ==="
python scripts/regenerate_train_data.py \
  --model Qwen3-1.7B \
  --concurrency 32 \
  --max-tokens 512 \
  --temperature 0.8 \
  --top-p 0.95 \
  --top-k 20 \
  --server-address 127.0.0.1:30000 \
  --input-file-path "$ROOT/data/sharegpt_30k/sharegpt_30k_eval.jsonl" \
  --output-file-path "$ROOT/data/regen/sharegpt_30k_eval_regen.jsonl"
echo "[$(date +%T)] eval done: $(wc -l < $ROOT/data/regen/sharegpt_30k_eval_regen.jsonl) rows"

echo "[$(date +%T)] === 停服务 ==="
kill "$(cat $ROOT/logs/regen_server.pid)" 2>/dev/null || true
sleep 8
nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo "[$(date +%T)] REGEN_ALL_DONE"
