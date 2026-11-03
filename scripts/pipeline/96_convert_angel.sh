ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
python - <<'PY'
import torch, json, os
from pathlib import Path
D = Path("/root/autodl-tmp/eagle3-qwen/exports/angelslim_eagle3")
sd = torch.load(D/"pytorch_model.bin", map_location="cpu", weights_only=True)
print("n_keys", len(sd))
for k, v in sd.items():
    print(f"  {k:38s} {tuple(v.shape)} {v.dtype}")
print("\ntied_word_embeddings in config:", json.load(open(D/"config.json"))["tie_word_embeddings"])

from safetensors.torch import save_file
out = {k: v.contiguous() for k, v in sd.items()}
save_file(out, str(D/"model.safetensors"))
print("saved model.safetensors", os.path.getsize(D/"model.safetensors")/1e6, "MB")
PY
