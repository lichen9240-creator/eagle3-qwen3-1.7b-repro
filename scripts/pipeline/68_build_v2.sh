set -e
ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
export HF_HOME=$ROOT/cache/huggingface
cd $ROOT/SpecForge

echo "[$(date +%T)] 过滤 ultrachat 生成结果"
python "$ROOT/repro/scripts/filter_zero_loss.py" \
  --model "$ROOT/models/Qwen3-1.7B" \
  --input "$ROOT/data/ultrachat_full/ultrachat_30k_regen.jsonl" \
  --output "$ROOT/data/ultrachat_full/ultrachat_filtered.jsonl" \
  --max-length 2048
wc -l $ROOT/data/ultrachat_full/ultrachat_filtered.jsonl

echo "[$(date +%T)] 合并 ShareGPT(含对端) + UltraChat"
python - <<'PY'
import json, random
from pathlib import Path
R = Path("/root/autodl-tmp/eagle3-qwen")

def load(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").split("\n") if l.strip()]

s = load(R/"data/merged/merged_filtered.jsonl")
u = load(R/"data/ultrachat_full/ultrachat_filtered.jsonl")
print(f"ShareGPT(含对端) 可用 {len(s)}")
print(f"UltraChat 可用    {len(u)}")

seen, allrows = set(), []
for x in s + u:
    k = str(x["id"])
    if k in seen:
        continue
    seen.add(k)
    allrows.append({"id": k, "conversations": x["conversations"]})
print(f"合并去重后 {len(allrows)}")

rng = random.Random(42)
rng.shuffle(allrows)
n_eval = round(len(allrows) * 0.05)
ev, tr = allrows[:n_eval], allrows[n_eval:]
print(f"切分: train {len(tr)} / eval {len(ev)}")

out = R/"data/v2"
out.mkdir(exist_ok=True)
for name, rows in [("train_v2.jsonl", tr), ("eval_v2.jsonl", ev)]:
    with open(out/name, "w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(name, len(rows))
PY
wc -l $ROOT/data/v2/*.jsonl
echo "[$(date +%T)] V2_DATA_READY"
