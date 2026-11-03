set -e
ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
CKPT=$(ls -d "$ROOT/runs/qwen3-1.7b-eagle3-42k"/epoch_* 2>/dev/null | sort -V | tail -1)
[ -n "$CKPT" ] || { echo "NO_CKPT"; exit 1; }
echo "CKPT=$CKPT"
DEST=$ROOT/exports/qwen3-1.7b-eagle3-v2-42k
rm -rf "$DEST"; mkdir -p "$DEST"
cp "$CKPT/config.json" "$DEST/config.json"
cp "$CKPT/model.safetensors" "$DEST/model.safetensors"
python - <<'PY'
import torch
from safetensors import safe_open
p = "/root/autodl-tmp/eagle3-qwen/exports/qwen3-1.7b-eagle3-v2-42k/model.safetensors"
with safe_open(p, framework="pt") as f:
    ks = list(f.keys())
    assert len(ks) == 15, len(ks)
    for k in ["fc.weight", "lm_head.weight", "d2t", "t2d", "norm.weight"]:
        assert k in ks, k
    d2t = f.get_tensor("d2t"); t2d = f.get_tensor("t2d")
    assert d2t.shape == (32000,) and t2d.shape == (151936,) and int(t2d.sum()) == 32000
    ids = d2t + torch.arange(32000, dtype=torch.int64)
    assert torch.all(ids >= 0) and torch.all(ids < 151936) and torch.all(ids[1:] > ids[:-1])
print("DRAFT_EXPORT_OK_v2")
PY
du -sh $DEST
