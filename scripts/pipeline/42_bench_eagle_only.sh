set -e
ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
export HF_HOME=$ROOT/cache/huggingface
cd $ROOT/SpecForge
echo "[$(date +%T)] EAGLE 吞吐测量（G10 未通过，仅诊断用）"
python "$ROOT/repro/scripts/bench_serving.py" \
  --url http://127.0.0.1:30000/generate \
  --data "$ROOT/data/bench/bench_90.jsonl" \
  --out-dir "$ROOT/runs/eagle" --tag eagle \
  --concurrency 1 4 8 --requests 90 --max-tokens 128 --warmup 3
echo "[$(date +%T)] EAGLE_TPUT_DONE"
