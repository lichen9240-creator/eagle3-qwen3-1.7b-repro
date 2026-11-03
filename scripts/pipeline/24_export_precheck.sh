ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
python - <<'PY'
import torch
from safetensors import safe_open
p = "/root/autodl-tmp/eagle3-qwen/runs/smoke_1k/epoch_0_step_1000/model.safetensors"
with safe_open(p, framework="pt") as f:
    ks = list(f.keys())
    print("n_keys", len(ks))
    for k in ["fc.weight", "lm_head.weight", "d2t", "t2d", "norm.weight"]:
        ok = k in ks
        print(f"  {k:16s} {'OK' if ok else 'MISSING'}", tuple(f.get_slice(k).get_shape()) if ok else "")
        assert ok, k
    d2t = f.get_tensor("d2t"); t2d = f.get_tensor("t2d")
    print("d2t", tuple(d2t.shape), "t2d", tuple(t2d.shape), "t2d.sum", int(t2d.sum()))
    assert d2t.shape == (32000,) and t2d.shape == (151936,) and int(t2d.sum()) == 32000
    ids = d2t + torch.arange(32000, dtype=torch.int64)
    assert torch.all(ids >= 0) and torch.all(ids < 151936) and torch.all(ids[1:] > ids[:-1])
    print("EXPORT_FORMAT_OK")
PY
