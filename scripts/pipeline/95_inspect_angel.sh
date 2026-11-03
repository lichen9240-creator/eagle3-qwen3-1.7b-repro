ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
D=$ROOT/exports/angelslim_eagle3
echo "=== 文件 ==="
ls -lh $D
echo "=== safetensors keys ==="
python - <<'PY'
from safetensors import safe_open
import glob
for p in glob.glob("/root/autodl-tmp/eagle3-qwen/exports/angelslim_eagle3/*.safetensors"):
    print("file:", p)
    with safe_open(p, framework="pt") as f:
        ks = list(f.keys())
        print("n_keys", len(ks))
        for k in ks:
            print(f"  {k:38s} {tuple(f.get_slice(k).get_shape())} {f.get_slice(k).get_dtype()}")
PY
