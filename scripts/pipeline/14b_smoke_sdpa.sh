set -e
ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
export CUDA_HOME=$ROOT/envs/cuda128
export CUDACXX=$CUDA_HOME/bin/nvcc
export PATH=$ROOT/envs/hostbin:$CUDA_HOME/bin:$PATH
export NVCC_PREPEND_FLAGS="-ccbin $ROOT/envs/hostbin/g++"
[ -d "$CUDA_HOME/lib" ] && ln -sfn "$CUDA_HOME/lib" "$CUDA_HOME/lib64"
export HF_HOME=$ROOT/cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $ROOT/SpecForge
rm -rf "$ROOT/runs/smoke_online_sdpa"
echo "[$(date +%T)] G6' 20 step smoke with ATTENTION=sdpa"
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
  scripts/train_eagle3.py \
  --target-model-path "$ROOT/models/Qwen3-1.7B" \
  --draft-model-config "$ROOT/repro/configs/qwen3-1.7b-eagle3.json" \
  --train-data-path "$ROOT/data/regen/smoke/train_100.jsonl" \
  --eval-data-path "$ROOT/data/regen/smoke/eval_100.jsonl" \
  --output-dir "$ROOT/runs/smoke_online_sdpa" \
  --num-epochs 1 --max-num-steps 20 \
  --batch-size 1 --learning-rate 1e-4 \
  --max-length 2048 --ttt-length 7 \
  --chat-template qwen3-instruct \
  --target-model-backend sglang \
  --attention-backend sdpa \
  --embedding-key model.embed_tokens.weight \
  --cache-dir "$ROOT/cache" \
  --save-interval 20 --eval-interval 20 --log-interval 1 \
  --seed 42 --report-to none \
  --sglang-mem-fraction-static 0.35 \
  --sglang-context-length 2048 \
  --dataloader-num-workers 0 --build-dataset-num-proc 8
echo "--- 产物 ---"
ls -d "$ROOT/runs/smoke_online_sdpa"/epoch_* 2>/dev/null || echo "NO_CKPT"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo "[$(date +%T)] SMOKE_SDPA_DONE"
