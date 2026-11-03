ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
python - <<'PY'
import aiohttp, requests
print("aiohttp", aiohttp.__version__)
print("requests", requests.__version__)
PY
echo "--- bench prompts ---"
head -1 $ROOT/repro/data/bench_prompts.jsonl | cut -c1-160
wc -l < $ROOT/repro/data/bench_prompts.jsonl
echo "--- 已生成的 bench_90 ---"
wc -l < $ROOT/data/bench/bench_90.jsonl
head -1 $ROOT/data/bench/bench_90.jsonl | cut -c1-120
