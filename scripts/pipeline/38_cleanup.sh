set -e
ROOT=/root/autodl-tmp/eagle3-qwen
RUN=$ROOT/runs/qwen3-1.7b-eagle3-30k

echo "=== 清理前 ==="
df -h / | tail -1
ls -d $RUN/epoch_* | sort -V | while read d; do echo "  $(basename $d)  $(du -sh $d | cut -f1)"; done

# 保留最后 2 个（最终 + 上一个备份），其余删除
ALL=$(ls -d $RUN/epoch_* | sort -V)
LAST=$(echo "$ALL" | tail -1)
PREV=$(echo "$ALL" | tail -2 | head -1)

echo "=== 校验保留项完整性 ==="
for d in "$LAST" "$PREV"; do
  echo "  $d"
  test -s "$d/model.safetensors" || { echo "    MISSING model.safetensors -> ABORT"; exit 1; }
  test -s "$d/config.json" || { echo "    MISSING config.json -> ABORT"; exit 1; }
  ls -lh "$d" | sed 's/^/    /'
done

echo "=== 删除早期 checkpoint ==="
echo "$ALL" | head -n -2 | while read d; do
  echo "  rm $(basename $d)"
  rm -rf "$d"
done

echo "=== 清理后 ==="
df -h / | tail -1
ls -d $RUN/epoch_* | sort -V
du -sh $RUN $ROOT/exports 2>/dev/null
