ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
echo "=== 官方 qwen3-8b-eagle3.json (SpecForge @961ca7c) ==="
cat $ROOT/SpecForge/configs/qwen3-8b-eagle3.json
echo
echo "=== 我们的 qwen3-1.7b-eagle3.json ==="
cat $ROOT/repro/configs/qwen3-1.7b-eagle3.json
echo
echo "=== Qwen3-1.7B target config ==="
python - <<'PY'
import json
c = json.load(open("/root/autodl-tmp/eagle3-qwen/models/Qwen3-1.7B/config.json"))
keys = ["hidden_size","num_hidden_layers","num_attention_heads","num_key_value_heads","head_dim",
        "intermediate_size","vocab_size","tie_word_embeddings","rope_theta","max_position_embeddings",
        "max_window_layers","rms_norm_eps","sliding_window","use_sliding_window","model_type"]
for k in keys:
    print(f"  {k:26s} {c.get(k)}")
PY
echo
echo "=== SpecForge 里还有哪些官方 eagle3 config ==="
ls $ROOT/SpecForge/configs/ | head -20
