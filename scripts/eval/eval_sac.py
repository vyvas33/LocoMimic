"""Evaluate a trained SAC checkpoint (native MuJoCo viewer or headless --eval).

macOS:  mjpython -m scripts.eval.eval_sac --ckpt <path> [--fast-sac]
Linux:  python    -m scripts.eval.eval_sac --ckpt <path> [--fast-sac]
"""

import argparse
import json
import os

import numpy as np
import torch

import mjlab.tasks  # noqa: F401
import src.tasks    # noqa: F401  registers Unitree-G1-Tracking

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer

from agents.configs.config_loader import load_config
from agents.sac.sac_agent import SACAgent
from rl.sac_env_wrapper import EmpiricalNormalization


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--algo', type=str, default='sac', choices=['sac', 'random'])
    parser.add_argument('--config', type=str, default='agents/configs/sac_config.yaml')
    parser.add_argument('--task', type=str, default='Unitree-G1-Tracking')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--num_envs', type=int, default=1)
    parser.add_argument('--fast-sac', action='store_true')
    parser.add_argument('--keep-terminations', action='store_true')
    parser.add_argument('--sampling-mode', type=str, default='start', choices=['start', 'uniform', 'adaptive'])
    parser.add_argument('--init-frame', type=int, default=None)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval-episodes', type=int, default=50)
    parser.add_argument('--json-out', type=str, default=None)
    args = parser.parse_args()

    if args.algo == 'sac' and args.ckpt is None:
        parser.error('--ckpt is required for --algo sac')

    config = load_config(args.config)
    config.device = args.device
    config.num_envs = args.num_envs
    if args.fast_sac:
        config.use_layer_norm = True
        config.use_mean_q = True
    if args.algo == 'sac':
        SACAgent.infer_config_from_ckpt(args.ckpt, config, args.device)

    configure_torch_backends()
    env_cfg = load_env_cfg(args.task, play=True)
    motion_cmd = env_cfg.commands['motion']
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.motion_file = config.motion_path
    motion_cmd.sampling_mode = args.sampling_mode
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.joint_position_range = (0.0, 0.0)

    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = config.seed
    if not args.eval and not args.keep_terminations:
        env_cfg.terminations = {}

    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)

    if args.init_frame is not None:
        mcmd = env.command_manager.get_term('motion')
        total = int(mcmd.motion.time_step_total)
        frame = max(0, min(args.init_frame, total - 1))

        def _fixed_sampling(env_ids, _mcmd=mcmd, _frame=frame):
            _mcmd.time_steps[env_ids] = _frame

        mcmd._uniform_sampling = _fixed_sampling
        mcmd._adaptive_sampling = _fixed_sampling
        original_resample = mcmd._resample_command

        def _resample_fixed(env_ids, _mcmd=mcmd, _orig=original_resample):
            _mcmd.cfg.sampling_mode = 'uniform'
            _orig(env_ids)

        mcmd._resample_command = _resample_fixed
        print(f'[INFO] init-frame = {frame} / {total}')

    env = RslRlVecEnvWrapper(env)
    inner = env.unwrapped
    actor_obs_dim = inner.single_observation_space.spaces['actor'].shape[0]
    critic_obs_dim = inner.single_observation_space.spaces['critic'].shape[0]
    act_dim = inner.single_action_space.shape[0]

    if args.algo == 'random':
        def policy(_obs_dict):
            return torch.empty(inner.num_envs, act_dim, device=args.device).uniform_(-1.0, 1.0)
    else:
        agent = SACAgent(actor_obs_dim, critic_obs_dim, act_dim, config)
        agent.load(args.ckpt)

        actor_rms = EmpiricalNormalization(actor_obs_dim, torch.device(args.device))
        rms_path = args.ckpt.replace('.pt', '.rms.pt')
        if os.path.exists(rms_path):
            sd = torch.load(rms_path, map_location=args.device)
            actor_rms.load_state_dict(sd['actor_rms'])
            print(f'Loaded obs normalizer: {rms_path}')
        else:
            print(f'WARNING: no .rms.pt sidecar at {rms_path}')

        @torch.no_grad()
        def policy(obs_dict):
            return agent.select_action(actor_rms(obs_dict['actor']), deterministic=True)

    if args.eval:
        run_eval(env, policy, args.eval_episodes, args.device, json_out=args.json_out, algo=args.algo)
    else:
        NativeMujocoViewer(env, policy).run()
    env.close()


@torch.no_grad()
def run_eval(env, policy, n_episodes: int, device: str, json_out: str | None = None, algo: str = 'sac'):
    num_envs = env.unwrapped.num_envs
    cur_ret = torch.zeros(num_envs, device=device)
    cur_len = torch.zeros(num_envs, dtype=torch.long, device=device)
    returns: list[float] = []
    lengths: list[int] = []

    obs_dict, _ = env.reset()
    while len(returns) < n_episodes:
        action = policy(obs_dict)
        obs_dict, rew, dones, _ = env.step(action)
        cur_ret += rew
        cur_len += 1
        for i in dones.nonzero(as_tuple=True)[0].tolist():
            returns.append(cur_ret[i].item())
            lengths.append(int(cur_len[i].item()))
            cur_ret[i] = 0
            cur_len[i] = 0
            if len(returns) >= n_episodes:
                break

    r = np.array(returns[:n_episodes], dtype=np.float64)
    l = np.array(lengths[:n_episodes], dtype=np.float64)
    print('\n========== eval results ==========')
    print(f'episodes:      {len(r)}')
    print(f'mean return:   {r.mean():.3f} ± {r.std():.3f}   (min {r.min():.3f}, max {r.max():.3f})')
    print(f'mean length:   {l.mean():.1f} ± {l.std():.1f}   (min {int(l.min())}, max {int(l.max())})')
    print('==================================')

    if json_out:
        out = {
            'algo': algo,
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
        os.makedirs(os.path.dirname(json_out) or '.', exist_ok=True)
        with open(json_out, 'w') as f:
            json.dump(out, f, indent=2)
        print(f'[INFO] wrote {json_out}')


if __name__ == '__main__':
    main()
