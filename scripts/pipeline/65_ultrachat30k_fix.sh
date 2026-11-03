set -e
ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
export HF_HOME=$ROOT/cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com
cd $ROOT/SpecForge
rm -rf $ROOT/data/ultrachat_full
echo "[$(date +%T)] 重新处理 ultrachat 35,000 条"
python scripts/prepare_data.py --dataset ultrachat \
  --output-path "$ROOT/data/ultrachat_full" --sample-size 35000 2>&1 | tail -5
wc -l $ROOT/data/ultrachat_full/*.jsonl

python - <<'PY'
import json, statistics
from pathlib import Path
R = Path("/root/autodl-tmp/eagle3-qwen")
src = R/"data/ultrachat_full/ultrachat_train.jsonl"

def valid(conv):
    if not conv or conv[0].get("role") != "user" or conv[-1].get("role") != "assistant":
        return None
    for i, m in enumerate(conv):
        if m.get("role") != ("user" if i % 2 == 0 else "assistant"):
            return None
        if not isinstance(m.get("content"), str) or not m["content"].strip():
            return None
    return len(conv) >= 2

rows = [json.loads(l) for l in src.read_text(encoding="utf-8").split("\n") if l.strip()]
ok = [r for r in rows if valid(r["conversations"])]
print(f"原始 {len(rows)} / 有效 {len(ok)}")
pick = ok[:30000]
out = R/"data/ultrachat_full/ultrachat_30k.jsonl"
with open(out, "w", encoding="utf-8") as f:
    for r in pick:
        f.write(json.dumps({"id": r["id"], "conversations": r["conversations"]}, ensure_ascii=False) + "\n")
chars = [sum(len(m["content"]) for m in r["conversations"]) for r in pick[:1000]]
print(f"取用 {len(pick)} 条, 字符中位数={statistics.median(chars):.0f}, 平均轮数={statistics.mean(len(r['conversations']) for r in pick[:1000]):.1f}")
PY
wc -l $ROOT/data/ultrachat_full/ultrachat_30k.jsonl
echo "[$(date +%T)] ULTRACHAT_30K_READY"
