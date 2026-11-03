ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
echo "=== 官方 Qwen3-8B online 示例脚本 ==="
cat $ROOT/SpecForge/examples/run_qwen3_8b_eagle3_online.sh
echo
echo "=== 官方示例里的关键超参 ==="
grep -E 'learning-rate|batch-size|num-epochs|max-length|ttt-length|num-steps|eagle-topk|num-draft-tokens|mem-fraction|attention-backend|target-model-backend' \
  $ROOT/SpecForge/examples/run_qwen3_8b_eagle3_online.sh
