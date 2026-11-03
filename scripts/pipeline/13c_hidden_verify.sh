ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
python - <<'PY'
import torch
from pathlib import Path

files = sorted(Path("/root/autodl-tmp/eagle3-qwen/data/regen/hidden_smoke").rglob("*.ckpt"))
print("ckpt 文件数:", len(files))
assert files, "no ckpt"

ok = True
dims = set()
for f in files:
    x = torch.load(f, map_location="cpu", weights_only=False)
    ks = sorted(x.keys())
    ids, mask = x["input_ids"], x["loss_mask"]
    h, a = x["hidden_state"], x["aux_hidden_state"]
    L = ids.shape[0]
    finite = bool(torch.isfinite(h.float()).all() and torch.isfinite(a.float()).all())
    good = (
        ks == ['aux_hidden_state', 'hidden_state', 'input_ids', 'loss_mask']
        and h.shape == (1, L, 2048)
        and a.shape == (1, L, 6144)
        and a.shape[-1] == 3 * h.shape[-1]
        and mask.shape == ids.shape
        and int(mask.sum()) > 0
        and finite
    )
    ok &= good
    dims.add((h.shape[-1], a.shape[-1]))
    print(f"{f.name:16s} len={L:5d} mask={int(mask.sum()):5d} finite={finite} {'OK' if good else 'FAIL'}")

print("维度集合:", dims)
assert dims == {(2048, 6144)}, "维度不符"
assert ok, "有样本不满足 schema"
print("HIDDEN_SCHEMA_OK")
PY
