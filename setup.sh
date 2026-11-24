#!/usr/bin/env bash
# Setup LocoMimic on Linux + CUDA. Run from repo root: bash setup.sh
set -euo pipefail

echo "==> [1/3] Initializing git submodules (unitree_rl_mjlab)"
git submodule update --init --recursive

echo "==> [2/3] Installing unitree_rl_mjlab and logging deps"
pip install -e ./unitree_rl_mjlab
pip install "mujoco==3.5.0" "warp-lang==1.12.0" wandb imageio matplotlib

echo "==> [3/3] Verifying CUDA + imports"
python - <<'PY'
import torch, mjlab
print(f"torch: {torch.__version__} | cuda: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device 0: {torch.cuda.get_device_name(0)}")
print(f"mjlab: {getattr(mjlab, '__version__', '?')}")
PY

echo
echo "Setup complete."
echo "Next steps:"
echo "  1. wandb login"
echo "  2. Convert motion CSV -> NPZ (if needed):"
echo "     python unitree_rl_mjlab/scripts/csv_to_npz.py \\"
echo "       --robot g1 --input-file data/lafan1_retargeted/g1/walk1_subject1.csv \\"
echo "       --output-name walk1_subject1.npz --device cuda:0"
echo "  3. Train SAC:"
echo "     python -m scripts.train.train_sac --name sac_vanilla --num_envs 4096"
