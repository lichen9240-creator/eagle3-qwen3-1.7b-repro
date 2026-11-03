set -e
ROOT=/root/autodl-tmp/eagle3-qwen
mkdir -p $ROOT/data/peer_regen
echo "dest dir ready: $ROOT/data/peer_regen"
df -h / | tail -1
ls -la $ROOT/data/peer_regen
