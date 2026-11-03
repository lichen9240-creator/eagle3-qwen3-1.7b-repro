ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
python - <<'PY'
import re, statistics
from pathlib import Path
p = Path("/root/autodl-tmp/eagle3-qwen/logs/train_30k.log")
txt = p.read_bytes().decode("utf-8", "replace").replace("\r", "\n")
pat = re.compile(r"(\d+)/21750 \[[\d:]+<[\d:?]+, *([\d.]+)it/s.*?loss=([-\d.eEnanNAN]+), acc=([\d.]+)")
rows = [(int(a), float(c), float(d)) for a, b, c, d in pat.findall(txt)]
print("样本数", len(rows))
if rows:
    losses = [r[1] for r in rows]
    print("最近 30 个 loss:", [round(x, 2) for x in losses[-30:]])
    print(f"loss: min={min(losses):.2f} max={max(losses):.2f} median={statistics.median(losses):.2f}")
    accs = [r[2] for r in rows]
    print(f"acc : min={min(accs):.2f} max={max(accs):.2f} median={statistics.median(accs):.2f}")
    n = max(1, len(rows) // 8)
    for i in range(0, len(rows), n):
        c = rows[i:i + n]
        print(f"  step {c[0][0]:>6}-{c[-1][0]:>6}: loss_mean={statistics.mean(x[1] for x in c):.2f} acc_mean={statistics.mean(x[2] for x in c):.2f}")
PY
