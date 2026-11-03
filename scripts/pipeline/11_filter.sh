set -e
ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
export CUDA_HOME=$ROOT/envs/cuda128
export CUDACXX=$CUDA_HOME/bin/nvcc
export PATH=$ROOT/envs/hostbin:$CUDA_HOME/bin:$PATH
export NVCC_PREPEND_FLAGS="-ccbin $ROOT/envs/hostbin/g++"
export HF_HOME=$ROOT/cache/huggingface
cd $ROOT/SpecForge

echo "[$(date +%T)] G3.5 过滤零 loss / 超长样本"
python "$ROOT/repro/scripts/filter_zero_loss.py" \
  --model "$ROOT/models/Qwen3-1.7B" \
  --input "$ROOT/data/regen/sharegpt_30k_train_regen.jsonl" \
  --output "$ROOT/data/regen/train_filtered.jsonl" \
  --max-length 2048
python "$ROOT/repro/scripts/filter_zero_loss.py" \
  --model "$ROOT/models/Qwen3-1.7B" \
  --input "$ROOT/data/regen/sharegpt_30k_eval_regen.jsonl" \
  --output "$ROOT/data/regen/eval_filtered.jsonl" \
  --max-length 2048

echo "--- G3 门禁：train>=20000, eval>=1000 ---"
wc -l $ROOT/data/regen/train_filtered.jsonl $ROOT/data/regen/eval_filtered.jsonl
echo "--- error 文件（若有） ---"
wc -l $ROOT/data/regen/*_regen_error.jsonl 2>/dev/null || echo "no error files"
echo "[$(date +%T)] FILTER_DONE"
