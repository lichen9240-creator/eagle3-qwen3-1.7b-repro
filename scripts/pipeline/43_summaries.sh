ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
python - <<'PY'
import json
from pathlib import Path
R = Path("/root/autodl-tmp/eagle3-qwen/runs")
print(f"{'case':<10} {'吞吐tok/s':>10} {'TTFT_ms':>9} {'TPOT_ms':>9} {'accept_len':>10} {'accept_rate':>11}")
res = {}
for tag in ["baseline", "eagle"]:
    for c in [1, 4, 8]:
        p = R / tag / f"{tag}_c{c}_summary.json"
        if not p.exists():
            continue
        d = json.load(open(p, encoding="utf-8"))
        al = d.get("spec_accept_length")
        ar = d.get("spec_accept_rate")
        al = al.get("mean") if isinstance(al, dict) else al
        ar = ar.get("mean") if isinstance(ar, dict) else ar
        res[(tag, c)] = d
        print(f"{tag+'_c'+str(c):<10} {d['throughput_tokens_s']:>10.1f} "
              f"{(d.get('ttft_ms') or 0):>9.1f} {(d.get('tpot_ms') or 0):>9.1f} "
              f"{(f'{al:.3f}' if al else '-'):>10} {(f'{ar:.3f}' if ar else '-'):>11}")
print()
for c in [1, 4, 8]:
    b, e = res[("baseline", c)]["throughput_tokens_s"], res[("eagle", c)]["throughput_tokens_s"]
    print(f"C={c}: baseline {b:.1f} -> eagle {e:.1f}  speedup = {e/b:.3f}x")
PY
