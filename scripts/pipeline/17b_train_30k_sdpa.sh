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
rm -rf "$ROOT/runs/qwen3-1.7b-eagle3-30k"
rm -rf "$ROOT/cache/processed_dataset" "$ROOT/cache/vocab_mapping"
echo "[$(date +%T)] G8 21,750 x 3 epoch 正式训练 (attention=sdpa)"
echo "train rows: $(wc -l < $ROOT/data/regen/train_filtered.jsonl)  eval rows: $(wc -l < $ROOT/data/regen/eval_filtered.jsonl)"
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
  scripts/train_eagle3.py \
  --target-model-path "$ROOT/models/Qwen3-1.7B" \
  --draft-model-config "$ROOT/repro/configs/qwen3-1.7b-eagle3.json" \
  --train-data-path "$ROOT/data/regen/train_filtered.jsonl" \
  --eval-data-path "$ROOT/data/regen/eval_filtered.jsonl" \
  --output-dir "$ROOT/runs/qwen3-1.7b-eagle3-30k" \
  --num-epochs 3 \
  --batch-size 1 --learning-rate 1e-4 \
  --max-length 2048 --ttt-length 7 \
  --chat-template qwen3-instruct \
  --target-model-backend sglang \
  --attention-backend sdpa \
  --embedding-key model.embed_tokens.weight \
  --cache-dir "$ROOT/cache" \
  --save-interval 5000 --eval-interval 10000 --log-interval 100 \
  --seed 42 --report-to none \
  --sglang-mem-fraction-static 0.35 \
  --sglang-context-length 2048 \
  --dataloader-num-workers 4 --build-dataset-num-proc 16
echo "--- checkpoints ---"
ls -d "$ROOT/runs/qwen3-1.7b-eagle3-30k"/epoch_* 2>/dev/null | sort -V | tail -6
echo "[$(date +%T)] G8_DONE"
