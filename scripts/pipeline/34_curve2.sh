ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
python - <<'PY'
import re, statistics
from pathlib import Path
txt = Path("/root/autodl-tmp/eagle3-qwen/logs/train_30k.log").read_bytes().decode("utf-8", "replace").replace("\r", "\n")
pat = re.compile(r"Training Epoch (\d+):.*?\|\s*(\d+)/21750 \[([\d:]+)<[\d:?]+, *([\d.]+)it/s.*?loss=([-\d.eEnanNAN]+), acc=([\d.]+)")
seen, rows = set(), []
for ep, step, el, rate, loss, acc in pat.findall(txt):
    k = (int(ep), int(step))
    if k in seen:
        continue
    seen.add(k)
    rows.append((int(ep), int(step), float(loss), float(acc), float(rate)))
print("唯一样本数", len(rows), "| epoch 分布:", {e: sum(1 for r in rows if r[0] == e) for e in sorted({r[0] for r in rows})})
print(f"\n{'epoch':>5s} {'step 区间':>13s} {'loss均值':>9s} {'acc均值':>8s} {'it/s':>6s}")
for ep in sorted({r[0] for r in rows}):
    er = [r for r in rows if r[0] == ep]
    er.sort(key=lambda x: x[1])
    n = max(1, len(er) // 4)
    for i in range(0, len(er), n):
        b = er[i:i + n]
        print(f"{ep:>5d} {b[0][1]:>6d}-{b[-1][1]:<6d} {statistics.mean(x[2] for x in b):>9.2f} "
              f"{statistics.mean(x[3] for x in b):>8.3f} {statistics.mean(x[4] for x in b):>6.2f}")
    print()
for ep in sorted({r[0] for r in rows}):
    er = [r for r in rows if r[0] == ep]
    head, tail = er[:300], er[-300:]
    print(f"epoch {ep}: 前300步 loss={statistics.mean(x[2] for x in head):.3f} acc={statistics.mean(x[3] for x in head):.3f}"
          f" | 后300步 loss={statistics.mean(x[2] for x in tail):.3f} acc={statistics.mean(x[3] for x in tail):.3f}")
PY
