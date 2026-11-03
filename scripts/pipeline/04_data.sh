set -e
ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
export HF_HOME=$ROOT/cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com
cd $ROOT/SpecForge

echo "[$(date +%T)] == G1.1 prepare_data sharegpt =="
python scripts/prepare_data.py --dataset sharegpt --output-path $ROOT/data/sharegpt_full 2>&1 | tail -15
echo "--- 产物 ---"
find $ROOT/data/sharegpt_full -maxdepth 1 -type f -printf '%f %s\n' | head
LINES=$(wc -l < $ROOT/data/sharegpt_full/sharegpt_train.jsonl)
echo "sharegpt_train.jsonl 行数 = $LINES"

echo "[$(date +%T)] == G1.2 固定 seed=42 切 30k / 95-5 =="
python $ROOT/repro/scripts/split_sharegpt.py \
  --input $ROOT/data/sharegpt_full/sharegpt_train.jsonl \
  --output-dir $ROOT/data/sharegpt_30k
echo "--- 门禁：必须 28500 / 1500 ---"
wc -l $ROOT/data/sharegpt_30k/*.jsonl
python - <<'PY'
import json, sys
from pathlib import Path
d = Path("/root/autodl-tmp/eagle3-qwen/data/sharegpt_30k")
tr = [json.loads(l) for l in (d/"sharegpt_30k_train.jsonl").read_text(encoding="utf-8").splitlines()]
ev = [json.loads(l) for l in (d/"sharegpt_30k_eval.jsonl").read_text(encoding="utf-8").splitlines()]
print("train", len(tr), "eval", len(ev))
assert len(tr) == 28500 and len(ev) == 1500, "数量不符"
idk = "id" if "id" in tr[0] else None
if idk:
    ids = [x[idk] for x in tr] + [x[idk] for x in ev]
    print("unique_ids", len(set(ids)), "total", len(ids))
    assert len(set(ids)) == len(ids), "ID 重复"
print("kept keys:", sorted(tr[0].keys())[:8])
print("G1_OK")
PY
echo "[$(date +%T)] DATA_DONE"
