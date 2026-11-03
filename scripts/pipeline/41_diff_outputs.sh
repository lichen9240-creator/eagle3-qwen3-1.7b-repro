ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
python - <<'PY'
import json
from pathlib import Path
R = Path("/root/autodl-tmp/eagle3-qwen/runs")
b = json.load(open(R/"baseline_correctness.json", encoding="utf-8"))
e = json.load(open(R/"eagle_correctness.json", encoding="utf-8"))
print(f"{'id':<12} {'match':>5s} {'len_b':>6s} {'len_e':>6s} {'共同前缀':>8s} {'首个不同位置':>12s}")
for x, y in zip(b, e):
    tb, te = x["text"] or "", y["text"] or ""
    common = 0
    for i in range(min(len(tb), len(te))):
        if tb[i] == te[i]:
            common += 1
        else:
            break
    match = tb == te
    print(f"{x['id']:<12} {str(match):>5s} {len(tb):>6d} {len(te):>6d} {common:>8d} "
          f"{('-' if match else common):>12}")
    if not match:
        i = common
        print(f"   base : ...{tb[max(0,i-30):i+50]!r}")
        print(f"   eagle: ...{te[max(0,i-30):i+50]!r}")
        mb = (x.get("meta") or {}); me = (y.get("meta") or {})
        print(f"   meta_b prompt={mb.get('prompt_tokens')} comp={mb.get('completion_tokens')} | "
              f"meta_e prompt={me.get('prompt_tokens')} comp={me.get('completion_tokens')} "
              f"accept={me.get('spec_accept_length')}")
PY
