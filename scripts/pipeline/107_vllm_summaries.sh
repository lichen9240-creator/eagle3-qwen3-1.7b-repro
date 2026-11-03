D=/root/autodl-tmp/eagle3-qwen/runs/vllm
for f in base_c1 base_c4 base1024_c1 eagle_c1 eagle_c4 eagle1024_c1; do
  echo "=== $f ==="
  cat $D/${f}_summary.json
  echo
done
