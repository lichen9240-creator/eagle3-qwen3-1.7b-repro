ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
python - <<'PY'
import json
from pathlib import Path
R = Path("/root/autodl-tmp/eagle3-qwen/runs")
def get(d, tag, c):
    p = R / d / tag / f"{tag}_c{c}_summary.json"
    if not p.exists(): return None
    j = json.load(open(p, encoding="utf-8"))
    al = j.get("spec_accept_length"); al = al.get("mean") if isinstance(al, dict) else al
    return j["throughput_tokens_s"], j["tpot_s"]["mean"]*1000, al

b1 = get("ksweep2", "baseline", 1); b4 = get("ksweep2", "baseline", 4)
print(f"baseline: C1={b1[0]:.1f} tok/s (TPOT {b1[1]:.2f}ms)  C4={b4[0]:.1f}\n")
print(f"{'case':<16} {'draft':<22} {'C1 tput':>8} {'speedup':>8} {'accept':>7} {'O':>6}")
rows = [
    ("ksweep2", "v2_k2", "ours v2 (fp16?no bf16)"),
    ("ksweep2", "v2_k3", "ours v2"),
    ("treesweep", "v2_t2_d8", "ours v2 tree"),
    ("angel", "angel_k2", "AngelSlim k=2"),
    ("angel", "angel_k3", "AngelSlim k=3"),
    ("angel", "angel_t2_d8", "AngelSlim tree"),
]
for d, t, desc in rows:
    a = get(d, t, 1)
    if not a:
        print(f"{t:<16} {desc:<22} missing"); continue
    sp = a[0]/b1[0]
    o = a[2]/sp if a[2] else None
    print(f"{t:<16} {desc:<22} {a[0]:>8.1f} {sp:>8.3f} {a[2]:>7.3f} {(f'{o:.3f}' if o else '-'):>6}")
PY
