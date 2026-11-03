set -e
ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
CKPT=$(ls -d "$ROOT/runs/qwen3-1.7b-eagle3-30k"/epoch_* 2>/dev/null | sort -V | tail -1)
[ -n "$CKPT" ] || { echo "NO_CKPT"; exit 1; }
echo "CKPT=$CKPT"
ls -lh "$CKPT"
rm -rf "$ROOT/exports/qwen3-1.7b-eagle3-sglang"
mkdir -p "$ROOT/exports/qwen3-1.7b-eagle3-sglang"
cp "$CKPT/config.json" "$ROOT/exports/qwen3-1.7b-eagle3-sglang/config.json"
cp "$CKPT/model.safetensors" "$ROOT/exports/qwen3-1.7b-eagle3-sglang/model.safetensors"

python - <<'PY'
import torch
from safetensors import safe_open
p = "/root/autodl-tmp/eagle3-qwen/exports/qwen3-1.7b-eagle3-sglang/model.safetensors"
with safe_open(p, framework="pt") as f:
    ks = list(f.keys())
    print("n_keys", len(ks))
    for k in ["fc.weight", "lm_head.weight", "d2t", "t2d", "norm.weight"]:
        assert k in ks, k
        print(k, tuple(f.get_slice(k).get_shape()), f.get_slice(k).get_dtype())
    d2t = f.get_tensor("d2t"); t2d = f.get_tensor("t2d")
    assert d2t.shape == (32000,)
    assert t2d.shape == (151936,)
    assert int(t2d.sum()) == 32000
    ids = d2t + torch.arange(32000, dtype=torch.int64)
    assert torch.all(ids >= 0) and torch.all(ids < 151936)
    assert torch.all(ids[1:] > ids[:-1])
    print("DRAFT_EXPORT_OK")
PY
echo "[$(date +%T)] G9_DONE"
