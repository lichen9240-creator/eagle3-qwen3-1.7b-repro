set -e
R=/root/autodl-tmp/eagle3-qwen
DST=/root/autodl-tmp/eagle3-qwen/data/peer_regen
cd $R/data/regen
echo "=== 源文件 md5 ==="
md5sum sharegpt_30k_train_regen.jsonl sharegpt_30k_eval_regen.jsonl | tee /tmp/peer_md5.txt
echo "=== 开始传输 ==="
time sshpass -p "${TARGET_SSH_PW:?export TARGET_SSH_PW}" scp -P 11570 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -o Compression=no \
  sharegpt_30k_train_regen.jsonl sharegpt_30k_eval_regen.jsonl \
  root@connect.westd.seetacloud.com:$DST/
echo "=== 完成 ==="
