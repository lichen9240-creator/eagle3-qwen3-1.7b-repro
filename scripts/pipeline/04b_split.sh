set -e
ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
echo "[$(date +%T)] G1.2 重新切分（先去重再按 seed=42 抽样）"
python $ROOT/repro/scripts/split_sharegpt.py \
  --input $ROOT/data/sharegpt_full/sharegpt_train.jsonl \
  --output-dir $ROOT/data/sharegpt_30k
echo "--- 门禁：必须 28500 / 1500 ---"
wc -l $ROOT/data/sharegpt_30k/*.jsonl
python - <<'PY'
import json
from pathlib import Path
d = Path("/root/autodl-tmp/eagle3-qwen/data/sharegpt_30k")
tr = [json.loads(l) for l in (d/"sharegpt_30k_train.jsonl").read_text(encoding="utf-8").splitlines()]
ev = [json.loads(l) for l in (d/"sharegpt_30k_eval.jsonl").read_text(encoding="utf-8").splitlines()]
print("train", len(tr), "eval", len(ev))
assert len(tr) == 28500 and len(ev) == 1500, "数量不符"
ids = [str(x["id"]) for x in tr] + [str(x["id"]) for x in ev]
print("unique_ids", len(set(ids)), "total_ids", len(ids))
assert len(set(ids)) == len(ids), "ID 重复"
assert not (set(str(x["id"]) for x in tr) & set(str(x["id"]) for x in ev)), "train/eval 交叉"
print("sample keys:", list(tr[0].keys()), "roles:", [m.get("role") for m in tr[0]["conversations"][:4]])
print("G1_OK")
PY
echo "[$(date +%T)] SPLIT_DONE"
