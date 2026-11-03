R=/root/autodl-tmp/eagle3-qwen
md5sum $R/data/sharegpt_30k/sharegpt_30k_train.jsonl $R/data/sharegpt_30k/sharegpt_30k_eval.jsonl
echo "--- 前3 id ---"
head -3 $R/data/sharegpt_30k/sharegpt_30k_train.jsonl | cut -c1-45
echo "--- 行数 ---"
wc -l $R/data/sharegpt_30k/*.jsonl $R/data/regen/train_filtered.jsonl $R/data/regen/eval_filtered.jsonl
