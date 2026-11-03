ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
echo "=== 官方 qwen3-4b-eagle3.json（第二个参照点）==="
cat $ROOT/SpecForge/configs/qwen3-4b-eagle3.json
echo
echo "=== 三方对照表 ==="
python - <<'PY'
import json
base = "/root/autodl-tmp/eagle3-qwen/"
c8 = json.load(open(base + "SpecForge/configs/qwen3-8b-eagle3.json"))
c4 = json.load(open(base + "SpecForge/configs/qwen3-4b-eagle3.json"))
c17 = json.load(open(base + "repro/configs/qwen3-1.7b-eagle3.json"))
t17 = json.load(open(base + "models/Qwen3-1.7B/config.json"))

fields = ["architectures","model_type","hidden_size","num_hidden_layers","num_attention_heads",
          "num_key_value_heads","head_dim","intermediate_size","vocab_size","draft_vocab_size",
          "tie_word_embeddings","rope_theta","max_position_embeddings","max_window_layers",
          "rms_norm_eps","sliding_window","use_sliding_window","attention_bias","use_cache"]
print(f"{'field':26s} {'8B官方':>22s} {'4B官方':>18s} {'1.7B(我们的)':>18s} {'1.7B target':>18s}")
for f in fields:
    def s(c):
        v = c.get(f, "-")
        return str(v) if not isinstance(v, list) else "/".join(v)
    print(f"{f:26s} {s(c8):>22s} {s(c4):>18s} {s(c17):>18s} {s(t17):>18s}")

same_as_8b = [f for f in fields if c8.get(f) == c17.get(f)]
diff = [f for f in fields if c8.get(f) != c17.get(f)]
print("\n与 8B 官方完全相同的字段:", len(same_as_8b))
print("不同的字段:", diff)
for f in diff:
    print(f"  {f}: 8B={c8.get(f)} -> ours={c17.get(f)} ; target={t17.get(f)} ; 是否跟随 target: {c17.get(f)==t17.get(f)}")
PY
echo
echo "=== 当前训练进度 ==="
tail -c 300 $ROOT/logs/train_30k.log | tr '\r' '\n' | grep 'Training Epoch' | tail -1 | sed 's/[^[:print:]]//g' | cut -c1-120
