R=/root/autodl-tmp/eagle3-qwen
echo "=== 收到的文件 ==="
ls -lh $R/data/peer_regen/
echo "=== md5 校验（应与源一致）==="
md5sum $R/data/peer_regen/*.jsonl
echo "期望: 542faa0f6ee608426416f4de5274ed74  train / 57c2469d5c877694f331f48d8a1e85e4  eval"
echo "=== 行数 ==="
wc -l $R/data/peer_regen/*.jsonl
