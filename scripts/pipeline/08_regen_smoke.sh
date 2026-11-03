set -e
ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
export HF_HOME=$ROOT/cache/huggingface
cd $ROOT/SpecForge
mkdir -p $ROOT/data/regen
echo "[$(date +%T)] G2: 100 条 regeneration smoke"
python scripts/regenerate_train_data.py \
  --model Qwen3-1.7B \
  --concurrency 32 \
  --max-tokens 512 \
  --temperature 0.8 \
  --top-p 0.95 \
  --top-k 20 \
  --server-address 127.0.0.1:30000 \
  --num-samples 100 \
  --input-file-path "$ROOT/data/sharegpt_30k/sharegpt_30k_train.jsonl" \
  --output-file-path "$ROOT/data/regen/smoke_100.jsonl"
echo "--- G2 门禁 ---"
wc -l $ROOT/data/regen/smoke_100.jsonl
if [ -s "$ROOT/data/regen/smoke_100_error.jsonl" ]; then echo "REGEN_ERROR_FILE_EXISTS"; head -3 "$ROOT/data/regen/smoke_100_error.jsonl"; exit 1; fi
echo "success 条数: $(grep -c '"status": *"success"' $ROOT/data/regen/smoke_100.jsonl || true)"
python - <<'PY'
import json
from pathlib import Path
p = Path("/root/autodl-tmp/eagle3-qwen/data/regen/smoke_100.jsonl")
rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
print("rows", len(rows), "keys", sorted(rows[0].keys())[:6])
empty = 0; think = 0
for r in rows:
    convs = r.get("conversations") or []
    txt = "\n".join(m.get("value","") if isinstance(m, dict) else str(m) for m in convs)
    if "<think>" in txt: think += 1
print("with_think:", think)
print("G2_OK" if len(rows) >= 95 and think == 0 else "G2_CHECK_NEEDED")
PY
echo "[$(date +%T)] G2_DONE"
