R=/root/autodl-tmp/eagle3-qwen
echo "=== PLAN.md 前 40 行 ==="
head -40 $R/PLAN.md 2>/dev/null
echo
echo "=== g1b_split.sh ==="
cat $R/g1b_split.sh 2>/dev/null | head -30
echo
echo "=== g3_regen_full.sh（关键：生成参数/模板）==="
cat $R/g3_regen_full.sh 2>/dev/null | head -40
echo
echo "=== g3b_filter.sh ==="
cat $R/g3b_filter.sh 2>/dev/null | head -20
