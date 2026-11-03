ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
python - <<'PY'
import json
from pathlib import Path
R = Path("/root/autodl-tmp/eagle3-qwen/runs/ksweep2")
cases = ["baseline", "v1_k2", "v1_k3", "v1_k4", "v2_k2", "v2_k3", "v2_k4"]
res = {}
for c in cases:
    res[c] = {}
    for k in [1, 4]:
        p = R / c / f"{c}_c{k}_summary.json"
        if not p.exists():
            res[c][k] = None; continue
        d = json.load(open(p, encoding="utf-8"))
        al = d.get("spec_accept_length"); al = al.get("mean") if isinstance(al, dict) else al
        res[c][k] = (d["throughput_tokens_s"], d["tpot_s"]["mean"]*1000, al)

print(f"{'case':<10} {'C':>2} {'tput':>8} {'speedup':>8} {'TPOT_ms':>8} {'accept':>7} {'O=τ/S':>6}")
for c in cases:
    for k in [1, 4]:
        v = res[c][k]
        if v is None:
            print(f"{c:<10} {k:>2}   missing"); continue
        tput, tpot, acc = v
        b = res["baseline"][k]
        sp = tput / b[0] if b and b[0] else None
        o = acc / sp if (acc and sp) else None
        print(f"{c:<10} {k:>2} {tput:>8.1f} {(f'{sp:.3f}x' if sp else '-'):>8} {tpot:>8.2f} "
              f"{(f'{acc:.3f}' if acc else '-'):>7} {(f'{o:.3f}' if o else '-'):>6}")
    print()
PY
