ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
cd $ROOT/SpecForge
python - <<'PY'
import torch
from transformers import LlamaConfig
from specforge.modeling.draft.llama3_eagle import LlamaForCausalLMEagle3

for tag, p in [("8B", "/root/autodl-tmp/eagle3-qwen/SpecForge/configs/qwen3-8b-eagle3.json"),
               ("1.7B", "/root/autodl-tmp/eagle3-qwen/repro/configs/qwen3-1.7b-eagle3.json")]:
    cfg = LlamaConfig.from_pretrained(p)
    with torch.device("meta"):
        m = LlamaForCausalLMEagle3(cfg)
    items = [(k, tuple(v.shape), v.requires_grad) for k, v in m.named_parameters()]
    print(f"=== {tag}: {len(items)} params ===")
    for k, s, rg in items[:3]:
        print(f"  {k:34s} {str(s):>18s} grad={rg}")
    print("  ...")
    print("  末 3 项:", [(k, s) for k, s, _ in items[-3:]])
    print("  含 embed:", any("embed" in k for k, _, _ in items))
PY
