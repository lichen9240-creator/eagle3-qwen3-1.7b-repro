ROOT=/root/autodl-tmp/eagle3-qwen
echo "=== 磁盘 ==="
df -h / | tail -1
echo "=== 内存 ==="
free -g | sed -n '2p'
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader
echo "=== 训练进程 ==="
ps -o pid,etime,pcpu,rss,cmd -p $(pgrep -f train_eagle3.py | head -1) 2>/dev/null | cut -c1-110
echo "=== dataloader workers ==="
pgrep -cf 'train_eagle3.py'
echo "=== 最近进度 ==="
tail -c 300 $ROOT/logs/train_30k.log | tr '\r' '\n' | grep 'Training Epoch' | tail -1 | sed 's/[^[:print:]]//g' | cut -c1-120
