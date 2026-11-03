cd /root/autodl-tmp/eagle3-qwen/exports
for f in draft-v1-sglang.tar.gz draft-v2-sglang.tar.gz; do
  if [ -f "$f" ]; then
    tar -xzf "$f" && echo "extracted $f"
  fi
done
ls -d */ 2>/dev/null
du -sh */ 2>/dev/null
echo "--- setup log ---"
tail -c 400 /root/autodl-tmp/eagle3-qwen/logs/setup.log | tail -4
echo "--- disk ---"
df -h / | tail -1
