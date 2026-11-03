ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
echo "=== 训练实时 ==="
tail -c 400 $ROOT/logs/train_30k.log | tr '\r' '\n' | grep 'Training Epoch' | tail -1 | sed 's/[^[:print:]]//g' | cut -c1-130
echo
echo "=== checkpoint 产物 ==="
ls -d $ROOT/runs/qwen3-1.7b-eagle3-30k/epoch_* 2>/dev/null | sort -V | while read d; do
  echo "  $(basename $d)  $(du -sh $d | cut -f1)"
done
echo
echo "=== 数据文件 ==="
ls -lh $ROOT/data/regen/train_filtered.jsonl $ROOT/data/regen/eval_filtered.jsonl | awk '{print "  "$9"  "$5}'
echo
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader
echo
echo "=== 进程 uptime ==="
ps -o etime= -p $(pgrep -f 'train_eagle3.py' | head -1) 2>/dev/null | tr -d ' '
