set -e
ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
export CUDA_HOME=$ROOT/envs/cuda128
export CUDACXX=$CUDA_HOME/bin/nvcc
export PATH=$ROOT/envs/hostbin:$CUDA_HOME/bin:$PATH
export NVCC_PREPEND_FLAGS="-ccbin $ROOT/envs/hostbin/g++"
[ -d "$CUDA_HOME/lib" ] && ln -sfn "$CUDA_HOME/lib" "$CUDA_HOME/lib64"
export HF_HOME=$ROOT/cache/huggingface
cd $ROOT/SpecForge
rm -rf "$ROOT/data/regen/hidden_smoke"
echo "[$(date +%T)] G5 10 条 hidden-state schema 检查"
torchrun --standalone --nproc_per_node=1 scripts/prepare_hidden_states.py \
  --target-model-path "$ROOT/models/Qwen3-1.7B" \
  --enable-aux-hidden-states \
  --data-path "$ROOT/data/regen/train_filtered.jsonl" \
  --output-path "$ROOT/data/regen/hidden_smoke" \
  --chat-template qwen3-instruct \
  --max-length 2048 \
  --tp-size 1 \
  --batch-size 1 \
  --num-samples 10 \
  --sglang-mem-fraction-static 0.35

python - <<'PY'
import torch
from pathlib import Path
files = sorted(Path("/root/autodl-tmp/eagle3-qwen/data/regen/hidden_smoke").rglob("*.ckpt"))
assert files, "no ckpt"
x = torch.load(files[0], map_location="cpu", weights_only=False)
print(sorted(x.keys()))
print("input_ids", tuple(x["input_ids"].shape))
print("loss_mask", tuple(x["loss_mask"].shape), int(x["loss_mask"].sum()))
print("hidden", tuple(x["hidden_state"].shape), x["hidden_state"].dtype)
print("aux", tuple(x["aux_hidden_state"].shape), x["aux_hidden_state"].dtype)
assert tuple(x["hidden_state"].shape[-2:]) == (2048, 2048)
assert tuple(x["aux_hidden_state"].shape[-2:]) == (2048, 6144)
assert int(x["loss_mask"].sum()) > 0
assert torch.isfinite(x["hidden_state"].float()).all()
assert torch.isfinite(x["aux_hidden_state"].float()).all()
print("HIDDEN_SCHEMA_OK")
PY
echo "[$(date +%T)] G5_DONE"
