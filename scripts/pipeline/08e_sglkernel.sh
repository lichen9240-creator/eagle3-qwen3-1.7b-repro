ROOT=/root/autodl-tmp/eagle3-qwen
PY=$ROOT/envs/eagle3/bin/python
echo "=== sgl_kernel 能否导入 ==="
$PY -c "
import sgl_kernel, torch
print('sgl_kernel ok', getattr(sgl_kernel,'__version__','?'))
for n in ['apply_rope_pos_ids_cos_sin_cache','fused_add_rmsnorm','rmsnorm']:
    print(n, hasattr(sgl_kernel, n))
" 2>&1 | tail -8
echo
echo "=== rotary_embedding.py 的分支条件 ==="
sed -n '25,60p;120,150p' $ROOT/envs/eagle3/lib/python3.12/site-packages/sglang/srt/layers/rotary_embedding.py
echo
echo "=== jit_kernel/rope.py 头部 ==="
sed -n '1,40p' $ROOT/envs/eagle3/lib/python3.12/site-packages/sglang/jit_kernel/rope.py 2>/dev/null
