ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
mkdir -p $ROOT/publish_extra
python - <<'PY'
import json, re
from pathlib import Path
R = Path("/root/autodl-tmp/eagle3-qwen")
out = R/"publish_extra"

# 1) 从训练日志抽完整曲线
txt = (R/"logs/train_30k.log").read_bytes().decode("utf-8", "replace").replace("\r", "\n")
pat = re.compile(r"Training Epoch (\d+):.*?\|\s*(\d+)/21750 \[([\d:]+)<[\d:?]+, *([\d.]+)it/s.*?loss=([-\d.eEnanNAN]+), acc=([\d.]+)")
seen, rows = set(), []
for ep, step, el, rate, loss, acc in pat.findall(txt):
    k = (int(ep), int(step))
    if k in seen:
        continue
    seen.add(k)
    rows.append((int(ep), int(step), float(loss), float(acc), float(rate)))
rows.sort()
with open(out/"training_curve_v1.csv", "w", encoding="utf-8") as f:
    f.write("epoch,step,loss,acc,it_per_s\n")
    for r in rows:
        f.write(",".join(str(x) for x in r) + "\n")
print("curve rows", len(rows))

# 2) 数据样例（各 3 条，内容截断到 300 字符）
def sample(src, dst, n=3, cut=300):
    lines = []
    with open(src, encoding="utf-8") as f:
        for i, l in enumerate(f):
            if i >= n: break
            d = json.loads(l)
            for m in d["conversations"]:
                if len(m["content"]) > cut:
                    m["content"] = m["content"][:cut] + "...(truncated)"
            lines.append(d)
    with open(dst, "w", encoding="utf-8") as f:
        for d in lines:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print("sample", dst.name, len(lines))

sample(R/"data/sharegpt_30k/sharegpt_30k_train.jsonl", out/"sample_sharegpt_split.jsonl")
sample(R/"data/regen/train_filtered.jsonl", out/"sample_sharegpt_regen_filtered.jsonl")
sample(R/"data/ultrachat_full/ultrachat_30k.jsonl", out/"sample_ultrachat_30k.jsonl")
PY
ls -la $ROOT/publish_extra/
