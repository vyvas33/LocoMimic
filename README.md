# LocoMimic

Whole-body motion imitation on the **Unitree G1** humanoid robot using retargeted LAFAN1 motions, focused on massively parallel GPU **Soft Actor-Critic (SAC)** variants on `unitree_rl_mjlab` (with PPO evaluation & comparison tooling).

> **Branches**
> - `main` — GPU-parallel `unitree_rl_mjlab` SAC (Vanilla SAC, FastSAC-style LayerNorm + Mean-Q, C51 Distributional SAC) + PPO evaluation/plotting.
> - `legacy-sac` — Phase 1 single-env CPU MuJoCo implementation (`LocoMimicEnv` + textbook SAC/PPO).

---

## Project Structure

```text
.
├── unitree_rl_mjlab/            # Submodule: GPU-vectorized mjlab env & rsl_rl PPO runner
├── agents/
│   ├── base.py
│   ├── configs/
│   │   ├── config_loader.py
│   │   ├── sac_config.yaml          # Vanilla SAC baseline
│   │   ├── sac_fast.yaml            # FastSAC-style (LayerNorm + mean-Q)
│   │   └── sac_distributional.yaml  # C51 distributional double critic
│   └── sac/
│       ├── actor.py
│       ├── critic.py
│       ├── distributional_critic.py
│       ├── replay_buffer.py
│       └── sac_agent.py
├── rl/
│   └── sac_env_wrapper.py       # Asymmetric actor/critic wrapper + EmpiricalNormalization
├── scripts/
│   ├── train/train_sac.py       # SAC training entry point
│   └── eval/
│       ├── eval_sac.py          # Interactive viewer or headless SAC/random eval
│       ├── eval_ppo.py          # Headless mjlab PPO eval (local or wandb:// ckpt)
│       ├── make_video.py        # Side-by-side reference vs. policy mp4/gif renderer
│       └── plot_training.py     # Shared env-step wandb training curves (PPO / SAC / FastSAC)
├── docs/sac_design.md           # Design notes & evolution log
├── data/                        # LAFAN1 retargeted CSVs & NPZ motions
└── setup.sh
```

---

## Setup (Linux, CUDA)

```bash
git clone --recurse-submodules https://github.com/vyvas33/LocoMimic.git
cd LocoMimic

conda create -n mjlab python=3.10 -y
conda activate mjlab

bash setup.sh
wandb login
```

### Converting LAFAN1 CSV to NPZ
`mjlab` consumes `.npz` motion files (50 Hz). Convert once from LAFAN1 CSV (30 Hz):
```bash
python unitree_rl_mjlab/scripts/csv_to_npz.py \
  --robot g1 \
  --input-file data/lafan1_retargeted/g1/walk1_subject1.csv \
  --output-name walk1_subject1.npz \
  --device cuda:0
```
Point `motion_path` in your chosen YAML config to the generated `.npz`.

---

## Training SAC

Run with `-m` from the repo root:

```bash
# Smoke test
python -m scripts.train.train_sac --name smoketest --num_envs 256 --iter 500

# Vanilla SAC
python -m scripts.train.train_sac \
  --name sac_vanilla \
  --config agents/configs/sac_config.yaml \
  --num_envs 4096

# FastSAC-style (LayerNorm + mean-of-Qs)
python -m scripts.train.train_sac \
  --name sac_fast \
  --config agents/configs/sac_fast.yaml \
  --num_envs 4096

# C51 Distributional SAC
python -m scripts.train.train_sac \
  --name sac_distq \
  --config agents/configs/sac_distributional.yaml \
  --num_envs 4096
```

CLI overrides: `--fast-sac`, `--distributional`, `--autotune` / `--no-autotune`, `--alpha-init X`, `--seed S`, `--iter N`, `--load <ckpt.pt>`.

Checkpoints are saved to `models/<run_name>/` alongside `<ckpt>.rms.pt` observation normalizer sidecars. See [docs/sac_design.md](docs/sac_design.md) for architecture and stability notes.

---

## Evaluation, Videos & Plots

```bash
# Interactive viewer (use `mjpython` on macOS, `python` on Linux)
python -m scripts.eval.eval_sac --ckpt models/<run_name>/final.pt --device cpu

# Headless SAC evaluation
python -m scripts.eval.eval_sac --ckpt models/<run_name>/final.pt --eval --eval-episodes 50

# Headless PPO evaluation (mjlab rsl_rl checkpoint)
python -m scripts.eval.eval_ppo --ckpt wandb://locomimic/locomimic/PPO_4096 --episodes 50 --num-envs 64

# Side-by-side reference vs. policy video (mp4 + gif)
python -m scripts.eval.make_video --algo sac --ckpt models/<run_name>/final.pt
python -m scripts.eval.make_video --algo ppo --ckpt wandb://locomimic/locomimic/PPO_4096

# Training curves from wandb
python -m scripts.eval.plot_training --max-steps 5e7
python -m scripts.eval.plot_training --mod1   # SAC vs FastSAC
```

*(Architecture flags like distributional Q and LayerNorm are automatically detected from the checkpoint.)*

---

## Acknowledgments

This project originated as a CMU 16-831 (Introduction to Robot Learning) course project built collaboratively with Dinesh Varun Shankar in [dineshvarunshankar/16831_IRL_Project](https://github.com/dineshvarunshankar/16831_IRL_Project), building on Unitree's [unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab) and design ideas from [FastSAC / HoloSoma](https://github.com/amazon-far/holosoma).
