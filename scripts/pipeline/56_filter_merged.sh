ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
export HF_HOME=$ROOT/cache/huggingface
cd $ROOT/SpecForge
echo "[$(date +%T)] 过滤合并后的 39,426 条"
python "$ROOT/repro/scripts/filter_zero_loss.py" \
  --model "$ROOT/models/Qwen3-1.7B" \
  --input "$ROOT/data/merged/merged_regen.jsonl" \
  --output "$ROOT/data/merged/merged_filtered.jsonl" \
  --max-length 2048
wc -l $ROOT/data/merged/merged_filtered.jsonl
echo "[$(date +%T)] FILTER_DONE"
