set -e
ROOT=/root/autodl-tmp/eagle3-qwen
cd $ROOT/exports
du -sh qwen3-1.7b-eagle3-sglang qwen3-1.7b-eagle3-v2-42k
tar -czf $ROOT/publish_v2/draft-v1-sglang.tar.gz -C $ROOT/exports qwen3-1.7b-eagle3-sglang
tar -czf $ROOT/publish_v2/draft-v2-sglang.tar.gz -C $ROOT/exports qwen3-1.7b-eagle3-v2-42k
ls -lh $ROOT/publish_v2/draft-*.tar.gz
md5sum $ROOT/publish_v2/draft-*.tar.gz
