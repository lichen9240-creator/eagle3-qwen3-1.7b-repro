ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
cd $ROOT/SpecForge
python - <<'PY'
import torch, json
from transformers import LlamaConfig
from specforge.modeling.draft.llama3_eagle import LlamaForCausalLMEagle3

def build(path):
    cfg = LlamaConfig.from_pretrained(path)
    with torch.device("meta"):
        m = LlamaForCausalLMEagle3(cfg)
    return {k: tuple(v.shape) for k, v in m.named_parameters()}

p8 = "/root/autodl-tmp/eagle3-qwen/SpecForge/configs/qwen3-8b-eagle3.json"
p17 = "/root/autodl-tmp/eagle3-qwen/repro/configs/qwen3-1.7b-eagle3.json"
s8, s17 = build(p8), build(p17)

print(f"{'param':38s} {'8B 官方':>22s} {'1.7B 我们':>20s}")
print("-" * 84)
for k in s8:
    print(f"{k:38s} {str(s8[k]):>22s} {str(s17.get(k, 'MISSING')):>20s}")

same_keys = set(s8) == set(s17)
print("\n参数名集合完全一致:", same_keys, "| 参数个数:", len(s8), len(s17))
n8 = sum(1 for v in s8.values() for _ in [0])
import math
def numel(shapes):
    n = 0
    for s in shapes.values():
        p = 1
        for d in s: p *= d
        n += p
    return n
print(f"参数量: 8B draft = {numel(s8)/1e6:.1f}M, 1.7B draft = {numel(s17)/1e6:.1f}M, 比例 = {numel(s8)/numel(s17):.2f}x")
PY
