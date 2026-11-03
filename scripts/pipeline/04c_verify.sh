set -e
ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
python - <<'PY'
import json
from pathlib import Path
d = Path("/root/autodl-tmp/eagle3-qwen/data/sharegpt_30k")
def load(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").split("\n") if l.strip()]
tr = load(d/"sharegpt_30k_train.jsonl"); ev = load(d/"sharegpt_30k_eval.jsonl")
print("train", len(tr), "eval", len(ev))
assert (len(tr), len(ev)) == (28500, 1500)
ids = [str(x["id"]) for x in tr] + [str(x["id"]) for x in ev]
print("unique", len(set(ids)), "total", len(ids))
assert len(set(ids)) == len(ids)
assert not ({str(x["id"]) for x in tr} & {str(x["id"]) for x in ev})
print("keys", list(tr[0].keys()))
print("roles", [m["role"] for m in tr[0]["conversations"]][:6])
print("turn_counts: min/%d max/%d" % (
    min(len(x["conversations"]) for x in tr), max(len(x["conversations"]) for x in tr)))
print("G1_OK")
PY
echo "[$(date +%T)] VERIFY_DONE"
