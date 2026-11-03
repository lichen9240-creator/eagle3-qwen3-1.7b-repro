set -e
ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
export HF_HOME=$ROOT/cache/huggingface
cd $ROOT/SpecForge
echo "[$(date +%T)] G4 config/词表/target head 审计"
python "$ROOT/repro/scripts/check_config.py" \
  --target-model "$ROOT/models/Qwen3-1.7B" \
  --draft-config "$ROOT/repro/configs/qwen3-1.7b-eagle3.json"
echo "[$(date +%T)] G4_DONE"
