ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
python - <<'PY'
import torch, json, shutil
from pathlib import Path
from safetensors.torch import load_file, save_file

A = Path("/root/autodl-tmp/eagle3-qwen/exports/angelslim_eagle3")
O = Path("/root/autodl-tmp/eagle3-qwen/exports/qwen3-1.7b-eagle3-v2-42k")

sa = load_file(str(A/"model.safetensors"))
so = load_file(str(O/"model.safetensors"))
print("=== d2t 前 8 个值 ===")
print("angel:", sa["d2t"][:8].tolist())
print("ours :", so["d2t"][:8].tolist())
print("=== d2t + arange 是否落在 [0,151936) ===")
for name, sd in [("angel", sa), ("ours", so)]:
    ids = sd["d2t"].to(torch.int64) + torch.arange(32000, dtype=torch.int64)
    print(f"  {name}: min={int(ids.min())} max={int(ids.max())} monotonic={bool(torch.all(ids[1:] > ids[:-1]))}")
print("=== d2t 尾部 5 个 ===")
print("angel:", sa["d2t"][-5:].tolist())
print("ours :", so["d2t"][-5:].tolist())

# 修正 tie_word_embeddings -> false，另存一份可加载的 draft
D = Path("/root/autodl-tmp/eagle3-qwen/exports/angelslim_eagle3_fix")
shutil.rmtree(D, ignore_errors=True)
D.mkdir(parents=True)
cfg = json.load(open(A/"config.json"))
cfg["tie_word_embeddings"] = False
json.dump(cfg, open(D/"config.json", "w"), indent=2)
# 转成 bf16，和目标模型 dtype 一致
sd_bf16 = {k: (v.to(torch.bfloat16) if v.dtype == torch.float16 else v) for k, v in sa.items()}
save_file(sd_bf16, str(D/"model.safetensors"))
print("wrote", D, "tie_word_embeddings=False, dtype bf16")
PY
