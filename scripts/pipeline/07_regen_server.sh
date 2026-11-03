set -e
ROOT=/root/autodl-tmp/eagle3-qwen
export PATH=$ROOT/envs/eagle3/bin:$PATH
export CUDA_HOME=$ROOT/envs/cuda128
export CUDACXX=$CUDA_HOME/bin/nvcc
export PATH=$CUDA_HOME/bin:$PATH
# 用 gcc-10 作为 nvcc 主机编译器（C++20 需要 >=10）
HB=$ROOT/envs/hostbin
mkdir -p $HB
for n in gcc g++ cc c++ gcc-10 g++-10; do
  src=/usr/bin/$n
  case "$n" in gcc|gcc-10) src=/usr/bin/gcc-10 ;; g++|g++-10) src=/usr/bin/g++-10 ;; cc) src=/usr/bin/gcc-10 ;; c++) src=/usr/bin/g++-10 ;; esac
  ln -sf $src $HB/$n
done
export PATH=$HB:$PATH
export CC=$HB/gcc
export CXX=$HB/g++
export NVCC_PREPEND_FLAGS="-ccbin $HB/g++"
# tvm_ffi 的 JIT 链接固定用 $CUDA_HOME/lib64
[ -d "$CUDA_HOME/lib" ] && ln -sfn "$CUDA_HOME/lib" "$CUDA_HOME/lib64"
export HF_HOME=$ROOT/cache/huggingface
cd $ROOT
if [ -f $ROOT/logs/regen_server.pid ]; then kill $(cat $ROOT/logs/regen_server.pid) 2>/dev/null || true; sleep 5; fi
nohup python -m sglang.launch_server \
  --model-path "$ROOT/models/Qwen3-1.7B" \
  --served-model-name Qwen3-1.7B \
  --chat-template "$ROOT/repro/templates/qwen3_nothink.jinja" \
  --dtype bfloat16 \
  --mem-fraction-static 0.70 \
  --max-running-requests 64 \
  --random-seed 42 \
  --host 0.0.0.0 --port 30000 \
  > "$ROOT/logs/regen_server.log" 2>&1 &
echo $! > "$ROOT/logs/regen_server.pid"
echo "[$(date +%T)] server pid=$(cat $ROOT/logs/regen_server.pid), 等待 /health"
for i in $(seq 1 120); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:30000/health || true)
  if [ "$code" = "200" ]; then echo "[$(date +%T)] HEALTH_200 after ${i}0s"; exit 0; fi
  sleep 10
done
echo "health timeout"; tail -40 "$ROOT/logs/regen_server.log"; exit 1
