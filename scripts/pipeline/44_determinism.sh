set -e
ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
export HF_HOME=$ROOT/cache/huggingface
cd $ROOT/SpecForge
echo "[$(date +%T)] EAGLE 第二次跑同样 10 条（确定性检验）"
python "$ROOT/repro/scripts/check_outputs.py" \
  --data "$ROOT/data/bench/bench_90.jsonl" \
  --out "$ROOT/runs/eagle_correctness_run2.json" \
  --n 10 --max-tokens 64

python - <<'PY'
import json
from pathlib import Path
R = Path("/root/autodl-tmp/eagle3-qwen/runs")
a = json.load(open(R/"eagle_correctness.json", encoding="utf-8"))
b = json.load(open(R/"eagle_correctness_run2.json", encoding="utf-8"))
same = sum(1 for x, y in zip(a, b) if x["text"] == y["text"])
print(f"EAGLE 两次运行自身一致: {same}/10")
for x, y in zip(a, b):
    if x["text"] != y["text"]:
        print("  DIFF", x["id"])
PY

echo "--- summary 字段 ---"
python - <<'PY'
import json
d = json.load(open("/root/autodl-tmp/eagle3-qwen/runs/eagle/eagle_c1_summary.json", encoding="utf-8"))
print(json.dumps(d, ensure_ascii=False, indent=1)[:1200])
PY
echo "[$(date +%T)] DETERMINISM_DONE"
