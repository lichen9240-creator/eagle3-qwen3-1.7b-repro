ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
python - <<'PY'
import re, statistics
from pathlib import Path
txt = Path("/root/autodl-tmp/eagle3-qwen/logs/train_30k.log").read_bytes().decode("utf-8", "replace").replace("\r", "\n")
pat = re.compile(r"(\d+)/21750 \[([\d:]+)<[\d:?]+, *([\d.]+)it/s.*?loss=([-\d.eEnanNAN]+), acc=([\d.]+)")
rows, seen = [], set()
for step, el, rate, loss, acc in pat.findall(txt):
    s = int(step)
    if s in seen:
        continue
    seen.add(s)
    rows.append((s, float(loss), float(acc), float(rate)))
rows.sort()
print("唯一样本数", len(rows), "step", rows[0][0], "->", rows[-1][0])
print(f"{'step 区间':>13s} {'loss均值':>9s} {'acc均值':>8s} {'it/s':>6s}")
n = 2500
buckets = {}
for s, l, a, r in rows:
    buckets.setdefault(s // n, []).append((s, l, a, r))
for k in sorted(buckets):
    b = buckets[k]
    print(f"{b[0][0]:>6d}-{b[-1][0]:<6d} {statistics.mean(x[1] for x in b):>9.2f} "
          f"{statistics.mean(x[2] for x in b):>8.3f} {statistics.mean(x[3] for x in b):>6.2f}")
# 首尾对比
head = rows[:500]; tail = rows[-500:]
print(f"\n前 500 步: loss={statistics.mean(x[1] for x in head):.3f} acc={statistics.mean(x[2] for x in head):.3f}")
print(f"后 500 步: loss={statistics.mean(x[1] for x in tail):.3f} acc={statistics.mean(x[2] for x in tail):.3f}")
PY
