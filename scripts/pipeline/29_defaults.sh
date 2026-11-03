ROOT=/root/autodl-tmp/eagle3-qwen
echo "=== train_eagle3.py 默认值（官方未显式传的参数）==="
grep -n -A3 '"--ttt-length"\|"--attention-backend"\|"--max-length"\|"--learning-rate"\|"--batch-size"\|"--warmup-ratio"\|"--max-grad-norm"' \
  $ROOT/SpecForge/scripts/train_eagle3.py | grep -E 'add_argument|default|help' | head -24
echo
echo "=== chat template 注册表里 qwen3-instruct vs qwen ==="
grep -n 'qwen3-instruct\|"qwen"' $ROOT/SpecForge/specforge/data/template.py | head -10
