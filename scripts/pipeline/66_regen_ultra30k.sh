set -e
ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
export HF_HOME=$ROOT/cache/huggingface
cd $ROOT/SpecForge
echo "[$(date +%T)] 生成 ultrachat 30,000 条（纯 target，concurrency 32）"
python scripts/regenerate_train_data.py \
  --model Qwen3-1.7B \
  --concurrency 32 \
  --max-tokens 512 \
  --temperature 0.8 \
  --top-p 0.95 \
  --top-k 20 \
  --server-address 127.0.0.1:30000 \
  --input-file-path "$ROOT/data/ultrachat_full/ultrachat_30k.jsonl" \
  --output-file-path "$ROOT/data/ultrachat_full/ultrachat_30k_regen.jsonl"
echo "[$(date +%T)] done: $(wc -l < $ROOT/data/ultrachat_full/ultrachat_30k_regen.jsonl) rows"
echo "[$(date +%T)] ULTRACHAT_30K_REGEN_DONE"
