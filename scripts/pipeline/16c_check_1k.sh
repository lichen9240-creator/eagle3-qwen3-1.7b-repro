set -e
ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
export HF_HOME=$ROOT/cache/huggingface
cd $ROOT/SpecForge
echo "[$(date +%T)] tokenize 90 条 benchmark"
mkdir -p $ROOT/data/bench
python "$ROOT/repro/scripts/tokenize_bench.py" \
  --model "$ROOT/models/Qwen3-1.7B" \
  --template "$ROOT/repro/templates/qwen3_nothink.jinja" \
  --input "$ROOT/repro/data/bench_prompts.jsonl" \
  --output "$ROOT/data/bench/bench_90.jsonl"
wc -l $ROOT/data/bench/bench_90.jsonl

echo "[$(date +%T)] G7 acceptance 初检 (3 条 x 32 token)"
python "$ROOT/repro/scripts/check_outputs.py" \
  --data "$ROOT/data/bench/bench_90.jsonl" \
  --out "$ROOT/runs/smoke_1k/correctness_3.json" \
  --n 3 --max-tokens 32

python - <<'PY'
import json
rows = json.load(open("/root/autodl-tmp/eagle3-qwen/runs/smoke_1k/correctness_3.json", encoding="utf-8"))
accs = []
for r in rows:
    m = r.get("meta") or {}
    a = m.get("spec_accept_length")
    accs.append(a)
    print(f"  id={r['id']:<12} accept_len={a} text[:40]={ (r.get('text') or '')[:40]!r}")
assert all(a is not None for a in accs), "spec_accept_length 缺失"
print("mean accept:", sum(accs)/len(accs))
print("G7_ACCEPT_OK" if sum(accs)/len(accs) > 1.0 else "G7_ACCEPT_TOO_LOW")
PY
echo "[$(date +%T)] G7_CHECK_DONE"
