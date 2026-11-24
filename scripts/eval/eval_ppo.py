"""Headless PPO eval using a local motion file.

usage:
  python -m scripts.eval.eval_ppo --ckpt wandb://locomimic/locomimic/PPO_4096 --episodes 50 --num-envs 64
"""

import argparse
import json
import os
from dataclasses import asdict

import numpy as np
import torch

import mjlab.tasks  # noqa: F401
import src.tasks    # noqa: F401  registers Unitree-G1-Tracking

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.torch import configure_torch_backends

from scripts.eval.make_video import _resolve_wandb_ckpt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True, help='local .pt path or wandb://entity/project/run[:file]')
    p.add_argument('--task', default='Unitree-G1-Tracking')
    p.add_argument('--motion-path', default='data/lafan1_retargeted_npz/g1/walk1_subject1.npz')
    p.add_argument('--device', default='cpu')
    p.add_argument('--num-envs', type=int, default=64)
    p.add_argument('--episodes', type=int, default=50)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--json-out', type=str, default=None)
    args = p.parse_args()

    configure_torch_backends()
    env_cfg = load_env_cfg(args.task, play=False)
    agent_cfg = load_rl_cfg(args.task)

    motion_cmd = env_cfg.commands.get('motion')
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.motion_file = args.motion_path
    motion_cmd.sampling_mode = 'start'

    env_cfg.observations['actor'].enable_corruption = True
    env_cfg.events.pop('push_robot', None)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed

    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=args.device), clip_actions=agent_cfg.clip_actions)

    ckpt_path = _resolve_wandb_ckpt(args.ckpt) if args.ckpt.startswith('wandb://') else args.ckpt
    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=args.device)
    runner.load(str(ckpt_path), map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)

    cur_ret = torch.zeros(args.num_envs, device=args.device)
    cur_len = torch.zeros(args.num_envs, dtype=torch.long, device=args.device)
    returns: list[float] = []
    lengths: list[int] = []

    motion_cmd = env.unwrapped.command_manager.get_term('motion')
    obs, _ = env.reset()
    # Populate relative body targets before the first termination check
    motion_cmd._update_command()
    obs = env.get_observations()

    while len(returns) < args.episodes:
        with torch.no_grad():
            action = policy(obs)
        obs, rew, dones, _ = env.step(action)
        cur_ret += rew
        cur_len += 1
        for i in dones.nonzero(as_tuple=True)[0].tolist():
            returns.append(cur_ret[i].item())
            lengths.append(int(cur_len[i].item()))
            cur_ret[i] = 0
            cur_len[i] = 0
            if len(returns) >= args.episodes:
                break

    env.close()

    r = np.array(returns[:args.episodes], dtype=np.float64)
    l = np.array(lengths[:args.episodes], dtype=np.float64)
    print('\n========== PPO eval results ==========')
    print(f'episodes:      {len(r)}')
    print(f'mean return:   {r.mean():.3f} ± {r.std():.3f}   (min {r.min():.3f}, max {r.max():.3f})')
    print(f'mean length:   {l.mean():.1f} ± {l.std():.1f}   (min {int(l.min())}, max {int(l.max())})')
    print('======================================')

    if args.json_out:
        out = {
            'algo': 'ppo',
            'n_episodes': int(len(r)),
            'return_mean': float(r.mean()),
            'return_std': float(r.std()),
            'return_min': float(r.min()),
            'return_max': float(r.max()),
            'length_mean': float(l.mean()),
            'length_std': float(l.std()),
            'length_min': int(l.min()),
            'length_max': int(l.max()),
        }
        os.makedirs(os.path.dirname(args.json_out) or '.', exist_ok=True)
        with open(args.json_out, 'w') as f:
            json.dump(out, f, indent=2)
        print(f'[INFO] wrote {args.json_out}')


if __name__ == '__main__':
    main()
