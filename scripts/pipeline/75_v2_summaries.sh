ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
python - <<'PY'
import json
from pathlib import Path
R = Path("/root/autodl-tmp/eagle3-qwen/runs")
def get(tag, c):
    d = json.load(open(R/tag/f"{tag}_c{c}_summary.json", encoding="utf-8"))
    al = d.get("spec_accept_length"); al = al.get("mean") if isinstance(al, dict) else al
    return d["throughput_tokens_s"], d["tpot_s"]["mean"]*1000, d["ttft_s"]["mean"]*1000, al

print(f"{'case':<12} {'tput':>8} {'TPOT_ms':>8} {'TTFT_ms':>8} {'accept':>7}")
for tag in ["baseline_v2", "eagle_v2"]:
    for c in [1,4,8]:
        t, tp, tf, al = get(tag, c)
        print(f"{tag+'_c'+str(c):<12} {t:>8.1f} {tp:>8.2f} {tf:>8.2f} {(f'{al:.3f}' if al else '-'):>7}")
print()
for c in [1,4,8]:
    b = get("baseline_v2", c)[0]; e = get("eagle_v2", c)[0]
    print(f"C={c}: {b:.1f} -> {e:.1f}  speedup = {e/b:.3f}x")
PY
