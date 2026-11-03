set -e
export PATH=/root/miniconda3/bin:$PATH
ROOT=/root/autodl-tmp/eagle3-qwen
mkdir -p $ROOT/{models,data/regen,runs,exports,logs,cache,envs}
cd $ROOT

echo "=== condarc ==="
conda config --show channels 2>/dev/null | head -5

echo "=== 创建 python3.12 环境 ==="
if [ -x $ROOT/envs/eagle3/bin/python ]; then
  echo "env exists: $($ROOT/envs/eagle3/bin/python -V)"
else
  conda create -y -p $ROOT/envs/eagle3 python=3.12 2>&1 | tail -5 \
  || conda create -y -p $ROOT/envs/eagle3 python=3.12 \
       -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main 2>&1 | tail -5
fi
$ROOT/envs/eagle3/bin/python -V

echo "=== clone SpecForge ==="
source /etc/network_turbo >/dev/null 2>&1
if [ ! -d $ROOT/SpecForge/.git ]; then
  rm -rf $ROOT/SpecForge
  git clone --filter=blob:none https://github.com/sgl-project/SpecForge.git $ROOT/SpecForge 2>&1 | tail -3
fi
cd $ROOT/SpecForge
git checkout 961ca7c3bb9d1d0a7aaf44db35e76e1f227a7be0 2>&1 | tail -3
git rev-parse HEAD | tee $ROOT/specforge_commit.txt
echo "--- pyproject 关键版本 ---"
grep -nE 'torch|transformers|sglang|python' pyproject.toml | head -30
