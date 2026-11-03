ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
mkdir -p $ROOT/data/merged
python - <<'PY'
import json
from pathlib import Path
R = Path("/root/autodl-tmp/eagle3-qwen")

def load(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").split("\n") if l.strip()]

ours_tr = load(R/"data/regen/sharegpt_30k_train_regen.jsonl")
ours_ev = load(R/"data/regen/sharegpt_30k_eval_regen.jsonl")
peer_tr = load(R/"data/peer_regen/sharegpt_30k_train_regen.jsonl")
peer_ev = load(R/"data/peer_regen/sharegpt_30k_eval_regen.jsonl")
print(f"ours train/eval = {len(ours_tr)}/{len(ours_ev)}")
print(f"peer train/eval = {len(peer_tr)}/{len(peer_ev)}")
print(f"总计 = {len(ours_tr)+len(ours_ev)+len(peer_tr)+len(peer_ev)}")

ours_ids = {str(x["id"]) for x in ours_tr + ours_ev}
peer_ids = {str(x["id"]) for x in peer_tr + peer_ev}
inter = ours_ids & peer_ids
print(f"\n我们的唯一 id: {len(ours_ids)}")
print(f"对端唯一 id  : {len(peer_ids)}")
print(f"重叠 id      : {len(inter)}  ({len(inter)/min(len(ours_ids),len(peer_ids))*100:.1f}% of 每批)")
print(f"合并去重后   : {len(ours_ids | peer_ids)}")

seen, merged = set(), []
for rows, tag in [(ours_tr, "ours"), (ours_ev, "ours"), (peer_tr, "peer"), (peer_ev, "peer")]:
    for x in rows:
        k = str(x["id"])
        if k in seen:
            continue
        seen.add(k)
        merged.append(x)
print(f"写入 merged_regen.jsonl: {len(merged)}")
with open(R/"data/merged/merged_regen.jsonl", "w", encoding="utf-8") as f:
    for x in merged:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")

json.dump(sorted(ours_ids), open(R/"data/merged/ours_ids.json", "w"))
json.dump(sorted(peer_ids), open(R/"data/merged/peer_ids.json", "w"))
PY
wc -l $ROOT/data/merged/merged_regen.jsonl
du -sh $ROOT/data/merged/merged_regen.jsonl
