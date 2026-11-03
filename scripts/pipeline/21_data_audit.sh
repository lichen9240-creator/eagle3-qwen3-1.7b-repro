ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
python - <<'PY'
import json, random, collections
from pathlib import Path
ROOT = Path("/root/autodl-tmp/eagle3-qwen")

def load(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").split("\n") if l.strip()]

raw = load(ROOT/"data/sharegpt_full/sharegpt_train.jsonl")
print("原始", len(raw))

seen, dedup = set(), []
for r in raw:
    k = str(r.get("id"))
    if k in seen: continue
    seen.add(k); dedup.append(r)
print("去重后", len(dedup))

def reason(r):
    roles = [m.get("role") for m in r.get("conversations", [])]
    if not roles: return "empty"
    if roles[0] == "assistant": return "starts_with_assistant"
    if roles[-1] != "assistant": return "ends_with_user"
    for i, rl in enumerate(roles):
        if rl != ("user" if i % 2 == 0 else "assistant"):
            return "non_alternating"
    return "ok"

c = collections.Counter(reason(r) for r in dedup)
print("丢弃原因分布:")
for k, v in c.most_common():
    print(f"  {k:24s} {v:7d}  {v/len(dedup)*100:5.1f}%")

tr = load(ROOT/"data/sharegpt_30k/sharegpt_30k_train.jsonl")
ev = load(ROOT/"data/sharegpt_30k/sharegpt_30k_eval.jsonl")
print(f"\n30k 切分: train={len(tr)} eval={len(ev)}")
print("轮数分布(train):", collections.Counter(len(r["conversations"])//2 for r in tr).most_common(6))
chars = [sum(len(m["content"]) for m in r["conversations"]) for r in tr]
chars.sort()
print(f"字符长度: p50={chars[len(chars)//2]} p90={chars[int(len(chars)*0.9)]} p99={chars[int(len(chars)*0.99)]} max={chars[-1]}")
PY
