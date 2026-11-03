# EAGLE-3 / Qwen3-1.7B 实验结果版本 v1

生成时间：2026-11-03
实例：pro-79360d2314fc（RTX 4090 48GB）

## 本版本训练产物
- 最终 draft checkpoint : runs/qwen3-1.7b-eagle3-30k/epoch_2_step_65250
- 备份 checkpoint       : runs/qwen3-1.7b-eagle3-30k/epoch_2_step_65000
- SGLang 可加载导出     : exports/qwen3-1.7b-eagle3-sglang
- 训练数据             : data/regen/train_filtered.jsonl (21,750)
- 验证数据             : data/regen/eval_filtered.jsonl  (1,154)
- benchmark 结果        : runs/baseline/, runs/eagle/

## v1 指标（后续对比基线）
- acceptance (C=1/4/8): 2.090 / 2.089 / 2.097
- throughput tok/s    : baseline 230.4 / 781.4 / 1464.0
                        EAGLE    256.9 / 843.3 / 1478.7
- speedup             : 1.115x / 1.079x / 1.010x
- 输出一致性          : 5/10（bf16 kernel 路径差异）
- 训练：65,250 步（21,750 x 3 epochs），loss 3.688 -> 0.89，acc 0.041 -> ~0.65

## 说明
- 本目录路径保持不变，避免破坏脚本引用。
- 后续新实验请另建目录或使用 v2 前缀，不要覆盖本版本。
