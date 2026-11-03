ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw --format=csv,noheader
echo "=== server 吞吐 ==="
grep -a 'Decode batch' $ROOT/logs/regen_server.log | tail -2 | sed 's/[^[:print:]]//g' | cut -c1-200
grep -a 'Prefill batch' $ROOT/logs/regen_server.log | tail -2 | sed 's/[^[:print:]]//g' | cut -c1-200
echo "=== 候选集长度分布 vs 已用数据 ==="
python - <<'PY'
import json, statistics
from pathlib import Path
R = Path("/root/autodl-tmp/eagle3-qwen")

def stats(p, n=800):
    rows = []
    with open(p, encoding="utf-8") as f:
        for i, l in enumerate(f):
            if i >= n: break
            rows.append(json.loads(l))
    turns = [len(r["conversations"]) for r in rows]
    chars = [sum(len(m["content"]) for m in r["conversations"]) for r in rows]
    return statistics.mean(turns), statistics.median(chars), max(chars)

for tag, p in [("已用 30k", R/"data/sharegpt_30k/sharegpt_30k_train.jsonl"),
               ("候选 29k", R/"data/pool/candidates.jsonl")]:
    t, c, mx = stats(p)
    print(f"{tag}: 平均轮数={t:.2f}  字符中位数={c:.0f}  最大={mx}")
PY
