# LocoMimic (Legacy — Custom MuJoCo Env)

Phase 1 single-environment CPU MuJoCo implementation of **LocoMimic** for Unitree G1 motion imitation from LAFAN1 CSV clips, containing the original textbook **SAC** (and PPO) pipeline.

> Looking for the current GPU-parallel `unitree_rl_mjlab` SAC (Vanilla / FastSAC / C51 Distributional SAC)? Switch to the **`main`** branch.

---

## Project Structure

```text
.
├── env/
│   ├── motion_clip.py         # Loads reference motion from LAFAN1 CSV
│   └── locomimic_env.py       # Custom Gymnasium MuJoCo environment
├── train/
│   ├── agents/
│   │   ├── base_agent.py
│   │   ├── sac/               # Vanilla single-env SAC (actor, double critic, agent)
│   │   └── ppo/               # Latest Phase-1 PPO implementation
│   ├── configs/
│   │   ├── config_loader.py
│   │   ├── sac_config.yaml
│   │   └── ppo_config.yaml
│   ├── replay_buffer.py
│   ├── train_sac.py
│   └── train_ppo.py
├── scripts/                   # Eval, comparison, plotting, and video rendering
├── tests/test_sac_gym.py
└── visualize.py
```

---

## Setup

```bash
conda create -n locomimic-legacy python=3.10 -y
conda activate locomimic-legacy

# Install PyTorch for your platform, then:
pip install -r requirements.txt

# Clone robot assets expected by LocoMimicEnv:
git clone https://github.com/google-deepmind/mujoco_menagerie
git clone https://github.com/unitreerobotics/unitree_mujoco
```

---

## Usage

```bash
# Train legacy SAC
PYTHONPATH=. python -m train.train_sac --name sac_legacy

# Train legacy PPO
PYTHONPATH=. python -m train.train_ppo --name ppo_legacy

# Evaluate & compare
python -m scripts.eval models/<run>/final.pt --algo sac --compare-random --save-plots
python -m scripts.render_comparison models/<run>/final.pt --algo sac

# Visualize reference motion
python visualize.py --csv data/lafan1_retargeted/g1/walk1_subject1.csv
```

See [scripts/README.md](scripts/README.md) for full evaluation and rendering options.

---

## Acknowledgments

This project originated as a CMU 16-831 (Introduction to Robot Learning) course project built collaboratively with Dinesh Varun Shankar in [dineshvarunshankar/16831_IRL_Project](https://github.com/dineshvarunshankar/16831_IRL_Project).
