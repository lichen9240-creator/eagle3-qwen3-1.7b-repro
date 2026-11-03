R=/root/autodl-tmp/eagle3-qwen
echo "=== 行数 ==="
wc -l $R/data/sharegpt_30k/*.jsonl $R/data/regen/*_regen.jsonl $R/data/regen/*_filtered.jsonl 2>/dev/null
echo "=== 切分文件 md5 ==="
md5sum $R/data/sharegpt_30k/sharegpt_30k_train.jsonl $R/data/sharegpt_30k/sharegpt_30k_eval.jsonl
echo "=== 前 3 个 id ==="
head -3 $R/data/sharegpt_30k/sharegpt_30k_train.jsonl | python3 -c "
import sys, json
for l in sys.stdin:
    d = json.loads(l); print(d['id'])
" 2>/dev/null || head -3 $R/data/sharegpt_30k/sharegpt_30k_train.jsonl | cut -c1-60
echo "=== filtered 前 3 个 id ==="
head -3 $R/data/regen/train_filtered.jsonl | cut -c1-60
echo "=== regen 一条样例结构 ==="
head -1 $R/data/regen/sharegpt_30k_train_regen.jsonl | cut -c1-400
echo "=== 他们的 regen 命令/日志 ==="
grep -ah 'regenerate_train_data.py' $R/logs/*.log 2>/dev/null | head -2 | cut -c1-300
echo "=== 是否有 repro/配置 ==="
ls $R/ 2>/dev/null
ls $R/repro/configs 2>/dev/null
