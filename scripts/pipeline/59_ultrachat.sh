set -e
ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
export HF_HOME=$ROOT/cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com
cd $ROOT/SpecForge
echo "[$(date +%T)] 下载 ultrachat 并转 schema（12000 条）"
python scripts/prepare_data.py --dataset ultrachat \
  --output-path "$ROOT/data/ultrachat_full" --sample-size 12000 2>&1 | tail -8
echo "--- 产物 ---"
wc -l $ROOT/data/ultrachat_full/*.jsonl 2>/dev/null
head -c 300 $ROOT/data/ultrachat_full/ultrachat_train.jsonl 2>/dev/null
echo
echo "[$(date +%T)] ULTRACHAT_DOWNLOAD_DONE"
