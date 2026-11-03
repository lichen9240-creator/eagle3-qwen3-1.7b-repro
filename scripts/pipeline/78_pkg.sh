set -e
ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
cd $ROOT
mkdir -p publish_v2
python - <<'PY'
import re
from pathlib import Path
R = Path("/root/autodl-tmp/eagle3-qwen")
txt = (R/"logs/train_v2.log").read_bytes().decode("utf-8", "replace").replace("\r", "\n")
pat = re.compile(r"Training Epoch (\d+):.*?\|\s*(\d+)/41405 \[[\d:]+<[\d:?]+, *([\d.]+)it/s.*?loss=([-\d.eEnanNAN]+), acc=([\d.]+)")
seen, rows = set(), []
for ep, step, rate, loss, acc in pat.findall(txt):
    k = (int(ep), int(step))
    if k in seen: continue
    seen.add(k)
    rows.append((int(ep), int(step), float(loss), float(acc), float(rate)))
rows.sort()
with open(R/"publish_v2/training_curve_v2.csv", "w", encoding="utf-8") as f:
    f.write("epoch,step,loss,acc,it_per_s\n")
    for r in rows: f.write(",".join(str(x) for x in r) + "\n")
print("v2 curve rows", len(rows))
PY
mkdir -p /tmp/pkg /tmp/pkg1
cp $ROOT/data/v2/train_v2.jsonl $ROOT/data/v2/eval_v2.jsonl $ROOT/data/bench/bench_90.jsonl /tmp/pkg/
cd /tmp/pkg && tar -czf $ROOT/publish_v2/eagle3-qwen3-1.7b-v2-data.tar.gz train_v2.jsonl eval_v2.jsonl bench_90.jsonl
rm -f /tmp/pkg/*.jsonl
cp $ROOT/data/regen/train_filtered.jsonl /tmp/pkg1/train_v1.jsonl
cp $ROOT/data/regen/eval_filtered.jsonl /tmp/pkg1/eval_v1.jsonl
cp $ROOT/data/bench/bench_90.jsonl /tmp/pkg1/bench_90.jsonl
cd /tmp/pkg1 && tar -czf $ROOT/publish_v2/eagle3-qwen3-1.7b-v1-data.tar.gz train_v1.jsonl eval_v1.jsonl bench_90.jsonl
rm -f /tmp/pkg1/*.jsonl
ls -lh $ROOT/publish_v2/
md5sum $ROOT/publish_v2/*.tar.gz
