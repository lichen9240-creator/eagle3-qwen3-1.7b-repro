set -x
echo "=== 杀掉所有 sglang 进程 ==="
pkill -f 'sglang.launch_server' || true
sleep 8
pgrep -fa sglang | head -5 || echo "none left"
echo "=== 端口占用 ==="
(ss -ltnp 2>/dev/null | grep 30000) || echo "port 30000 free"
echo "=== 删除 EAGLE3 生成的脏数据 ==="
rm -f /root/autodl-tmp/eagle3-qwen/data/pool/candidates_regen.jsonl \
      /root/autodl-tmp/eagle3-qwen/data/ultrachat_full/ultrachat_10k_regen.jsonl
ls -la /root/autodl-tmp/eagle3-qwen/data/pool/ 2>/dev/null
echo "=== 杀掉两个 regen 客户端 ==="
pkill -f 'regenerate_train_data.py' || true
sleep 3
pgrep -cf regenerate_train_data || echo "0 clients"
