echo "=== 磁盘/GPU ==="
df -h / | tail -1
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader 2>/dev/null | head -1
echo "=== /root/autodl-tmp ==="
ls -la /root/autodl-tmp 2>/dev/null | head -20
echo "=== 找数据文件 ==="
find /root/autodl-tmp -maxdepth 4 \( -name '*regen*.jsonl' -o -name '*sharegpt*30k*' -o -name '*filtered*.jsonl' \) -printf '%12s  %p\n' 2>/dev/null | head -25
echo "=== 找目录 ==="
du -sh /root/autodl-tmp/* 2>/dev/null | sort -h | tail -10
