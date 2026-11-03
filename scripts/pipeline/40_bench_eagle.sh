set -e
ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
export HF_HOME=$ROOT/cache/huggingface
cd $ROOT/SpecForge
echo "[$(date +%T)] G10 eagle correctness (10 x 64)"
python "$ROOT/repro/scripts/check_outputs.py" \
  --data "$ROOT/data/bench/bench_90.jsonl" \
  --out "$ROOT/runs/eagle_correctness.json" \
  --n 10 --max-tokens 64
echo "[$(date +%T)] G10 一致性比对 baseline vs eagle"
python "$ROOT/repro/scripts/compare_outputs.py" \
  --baseline "$ROOT/runs/baseline_correctness.json" \
  --eagle "$ROOT/runs/eagle_correctness.json"
echo "[$(date +%T)] G11/G12 eagle throughput C=1,4,8"
python "$ROOT/repro/scripts/bench_serving.py" \
  --url http://127.0.0.1:30000/generate \
  --data "$ROOT/data/bench/bench_90.jsonl" \
  --out-dir "$ROOT/runs/eagle" --tag eagle \
  --concurrency 1 4 8 --requests 90 --max-tokens 128 --warmup 3
echo "[$(date +%T)] BENCH_EAGLE_DONE"
