ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
cd $ROOT/SpecForge
# 抽 2000 条 train + 300 条 eval 做真实的 filter 预估
head -n 2000 $ROOT/data/sharegpt_30k/sharegpt_30k_train.jsonl > /tmp/sample_train.jsonl
python "$ROOT/repro/scripts/filter_zero_loss.py" \
  --model "$ROOT/models/Qwen3-1.7B" \
  --input /tmp/sample_train.jsonl \
  --output /tmp/sample_train_kept.jsonl \
  --max-length 2048
echo "--- 截断后 token 长度分布（保留样本）---"
python - <<'PY'
import json
from transformers import AutoTokenizer
from specforge.data.template import TEMPLATE_REGISTRY
tok = AutoTokenizer.from_pretrained("/root/autodl-tmp/eagle3-qwen/models/Qwen3-1.7B")
tpl = TEMPLATE_REGISTRY.get("qwen3-instruct")
rows = [json.loads(l) for l in open("/tmp/sample_train_kept.jsonl", encoding="utf-8")]
lens = []
for r in rows[:800]:
    text = tpl.render(r["conversations"], add_generation_prompt=False) if hasattr(tpl, "render") else None
    ids = tok.apply_chat_template(r["conversations"], tokenize=True, add_generation_prompt=False)
    lens.append(len(ids))
lens.sort()
n = len(lens)
print(f"样本 {n} 条，原始 token 长度: p50={lens[n//2]} p90={lens[int(n*.9)]} p99={lens[int(n*.99)]} max={lens[-1]}")
print(f"超过 2048 的比例: {sum(1 for x in lens if x > 2048)/n*100:.1f}%")
PY
