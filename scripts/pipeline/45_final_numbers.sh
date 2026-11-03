ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
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

print()
for tag in ["baseline", "eagle"]:
    d = json.load(open(R/tag/f"{tag}_c1_summary.json", encoding="utf-8"))
    t = d["ttft_s"]["mean"] * 1000
    p = d["tpot_s"]["mean"] * 1000
    print(f"{tag}_c1: tput={d['throughput_tokens_s']:.1f} tok/s  TTFT={t:.2f}ms  TPOT={p:.2f}ms  "
          f"latency={d['latency_s']['mean']*1000:.0f}ms  out_tokens={d['output_tokens']}")
PY
