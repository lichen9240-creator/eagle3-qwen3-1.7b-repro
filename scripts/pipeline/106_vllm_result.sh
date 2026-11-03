L=/root/autodl-tmp/eagle3-qwen/logs/vllm_eagle2.log
grep -a '"conc"' $L | tail -6
echo "--- done? ---"
grep -ac VLLM_EAGLE_DONE $L
echo "--- 最新 acceptance ---"
grep -a 'Mean acceptance length' $L | tail -2 | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-200
