ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
mkdir -p $ROOT/data/pool
python - <<'PY'
import json
from pathlib import Path
R = Path("/root/autodl-tmp/eagle3-qwen")

def load(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").split("\n") if l.strip()]

raw = load(R/"data/sharegpt_full/sharegpt_train.jsonl")
seen, dedup = set(), []
for r in raw:
    k = str(r["id"])
    if k in seen:
        continue
    seen.add(k)
    dedup.append(r)
print("原始", len(raw), "去重", len(dedup))

used = set(json.load(open(R/"data/merged/ours_ids.json"))) | set(json.load(open(R/"data/merged/peer_ids.json")))
print("已用 id", len(used))

def recover(conv):
    """掐掉开头非 user、结尾非 assistant 的消息，再要求 user/assistant 严格交替。"""
    msgs = list(conv)
    while msgs and msgs[0].get("role") != "user":
        msgs.pop(0)
    while msgs and msgs[-1].get("role") != "assistant":
        msgs.pop()
    if len(msgs) < 2:
        return None
    for i, m in enumerate(msgs):
        if m.get("role") != ("user" if i % 2 == 0 else "assistant"):
            return None
        if not isinstance(m.get("content"), str) or not m["content"].strip():
            return None
    return msgs

cand, stats = [], {"already_used": 0, "no_conv": 0, "empty": 0, "non_alt": 0, "ok": 0, "recovered": 0}
for r in dedup:
    k = str(r["id"])
    if k in used:
        stats["already_used"] += 1
        continue
    conv = r.get("conversations") or []
    if not conv:
        stats["no_conv"] += 1
        continue
    was_assistant_first = conv[0].get("role") == "assistant"
    msgs = recover(conv)
    if msgs is None:
        # 区分：是因为交替性不行，还是本身为空
        stats["non_alt"] += 1
        continue
    stats["ok"] += 1
    if was_assistant_first:
        stats["recovered"] += 1
    cand.append({"id": k, "conversations": msgs})

print("\n候选统计:")
for k, v in stats.items():
    print(f"  {k:16s} {v}")
print(f"\n写入 candidates: {len(cand)} 条")
with open(R/"data/pool/candidates.jsonl", "w", encoding="utf-8") as f:
    for x in cand:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")

# 估算过滤后可用量
print(f"\n预计过滤后可用 ≈ {len(cand)*0.765:.0f} 条")
print(f"加上现有 30,149 → 约 {30149 + len(cand)*0.765:.0f} 条")
PY
wc -l $ROOT/data/pool/candidates.jsonl
du -sh $ROOT/data/pool/candidates.jsonl
