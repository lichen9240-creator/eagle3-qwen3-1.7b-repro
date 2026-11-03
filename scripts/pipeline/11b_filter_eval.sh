set -e
ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
cd $ROOT/SpecForge
echo "[$(date +%T)] 过滤 eval"
python "$ROOT/repro/scripts/filter_zero_loss.py" \
  --model "$ROOT/models/Qwen3-1.7B" \
  --input "$ROOT/data/regen/sharegpt_30k_eval_regen.jsonl" \
  --output "$ROOT/data/regen/eval_filtered.jsonl" \
  --max-length 2048
wc -l $ROOT/data/regen/train_filtered.jsonl $ROOT/data/regen/eval_filtered.jsonl
echo "[$(date +%T)] G3_GATE_CHECK"
