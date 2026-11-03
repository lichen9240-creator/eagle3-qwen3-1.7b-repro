ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
python - <<'PY'
import json
from pathlib import Path
R = Path("/root/autodl-tmp/eagle3-qwen/data/v2")

def load(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").split("\n") if l.strip()]

tr = load(R/"train_v2.jsonl"); ev = load(R/"eval_v2.jsonl")
print(f"train {len(tr)}  eval {len(ev)}")
ids = [str(x["id"]) for x in tr] + [str(x["id"]) for x in ev]
print("唯一 id:", len(set(ids)), "总:", len(ids))
assert len(set(ids)) == len(ids), "有重复 id"
assert not ({str(x["id"]) for x in tr} & {str(x["id"]) for x in ev}), "train/eval 交叉"
for x in tr[:200] + ev[:50]:
    c = x["conversations"]
    assert c[0]["role"] == "user" and c[-1]["role"] == "assistant", "结构错误"
    assert all(m["role"] == ("user" if i % 2 == 0 else "assistant") for i, m in enumerate(c))
print("结构校验通过")
import statistics
chars = [sum(len(m["content"]) for m in x["conversations"]) for x in tr[:2000]]
print(f"train 字符中位数 {statistics.median(chars):.0f}, 平均轮数 {statistics.mean(len(x['conversations']) for x in tr[:2000]):.1f}")
print("V2_DATA_OK")
PY
