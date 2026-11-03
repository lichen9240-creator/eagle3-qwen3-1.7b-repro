ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
python - <<'PY'
import json
from pathlib import Path
d = Path("/root/autodl-tmp/eagle3-qwen/data/sharegpt_30k")
def load(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").split("\n") if l.strip()]
for name in ["sharegpt_30k_train.jsonl", "sharegpt_30k_eval.jsonl"]:
    rows = load(d/name)
    bad_first = sum(1 for r in rows if (r["conversations"][0]["role"] != "user"))
    # 角色必须严格交替 user/assistant 且以 user 开头、以 assistant 结尾
    def ok(r):
        roles = [m["role"] for m in r["conversations"]]
        if roles[0] != "user" or roles[-1] != "assistant":
            return False
        for i, rl in enumerate(roles):
            if rl != ("user" if i % 2 == 0 else "assistant"):
                return False
        return True
    good = sum(1 for r in rows if ok(r))
    print(f"{name}: total={len(rows)} starts_with_assistant={bad_first} ({bad_first/len(rows)*100:.1f}%) strictly_alternating={good} ({good/len(rows)*100:.1f}%)")
PY
echo "--- smoke error 分布 ---"
python - <<'PY'
import json, collections
from pathlib import Path
p = Path("/root/autodl-tmp/eagle3-qwen/data/regen/smoke_100_error.jsonl")
c = collections.Counter()
for l in p.read_text(encoding="utf-8").split("\n"):
    if l.strip():
        c[json.loads(l).get("error")] += 1
for k, v in c.most_common():
    print(f"  {v:4d}  {k}")
PY
