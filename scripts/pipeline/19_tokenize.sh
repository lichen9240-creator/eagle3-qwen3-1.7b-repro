set -e
ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
export HF_HOME=$ROOT/cache/huggingface
cd $ROOT/SpecForge
mkdir -p $ROOT/data/bench
echo "[$(date +%T)] tokenize 90 条固定 benchmark prompt"
python "$ROOT/repro/scripts/tokenize_bench.py" \
  --model "$ROOT/models/Qwen3-1.7B" \
  --template "$ROOT/repro/templates/qwen3_nothink.jinja" \
  --input "$ROOT/repro/data/bench_prompts.jsonl" \
  --output "$ROOT/data/bench/bench_90.jsonl"
wc -l $ROOT/data/bench/bench_90.jsonl
echo "[$(date +%T)] TOKENIZE_DONE"
