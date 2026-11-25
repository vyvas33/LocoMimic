"""Unified fixed-config evaluation suite for Random / SAC / PPO policies.

Evaluates one or more models under identical env configs (nominal & perturbed)
and saves JSON + Markdown summary table.

usage:
  python -m scripts.eval.eval_suite \
    --models random:random:none \
             sac_fast:sac:wandb://GenAi-hw/locomimic/ze3j2e5k:final.pt \
             ppo:ppo:wandb://GenAi-hw/locomimic/PPO_RUN \
    --suites nominal perturbed --episodes 100 --num-envs 64 --device cuda:0
"""

import argparse
import json
import os
import re
from dataclasses import asdict

import numpy as np
import torch

import mjlab.tasks  # noqa: F401
import src.tasks    # noqa: F401

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.lab_api.math import quat_error_magnitude
from mjlab.utils.torch import configure_torch_backends

from agents.configs.config_loader import load_config
from agents.sac.sac_agent import SACAgent
from rl.sac_env_wrapper import EmpiricalNormalization


def resolve_wandb_file(uri: str, cache_dir: str = 'outputs/wandb_ckpts') -> tuple[str, str | None]:
    """Download wandb://entity/project/run_id_or_name[:filename] (+ .rms.pt sidecar if present)."""
    import wandb

    rest = uri.removeprefix('wandb://')
    run_path, filename = rest.split(':', 1) if ':' in rest else (rest, None)
    entity, project, run_ref = run_path.split('/')

    api = wandb.Api(timeout=60)
    try:
        run = api.run(f'{entity}/{project}/{run_ref}')
    except Exception:
        runs = list(api.runs(f'{entity}/{project}', filters={'display_name': run_ref}))
        if not runs:
            raise ValueError(f'no run {run_ref!r} in {entity}/{project}')
        run = runs[0]

    files = {f.name: f for f in run.files()}
    if filename is None:
        ckpts = [n for n in files if re.fullmatch(r'model_\d+\.pt', n)]
        if not ckpts:
            raise ValueError(f'no model_*.pt in {run_ref!r}')
        ckpts.sort(key=lambda n: int(re.search(r'\d+', n).group()))
        filename = ckpts[-1]
    if filename not in files:
        raise ValueError(f'{filename!r} not found in {run_ref!r}')

    local_dir = os.path.join(cache_dir, run.id)
    os.makedirs(local_dir, exist_ok=True)
    local_ckpt = os.path.join(local_dir, filename)
    if not os.path.exists(local_ckpt):
        files[filename].download(root=local_dir, replace=False)

    rms_name = filename.replace('.pt', '.rms.pt')
    local_rms = None
    if rms_name in files:
        local_rms = os.path.join(local_dir, rms_name)
        if not os.path.exists(local_rms):
            files[rms_name].download(root=local_dir, replace=False)

    return local_ckpt, local_rms


def build_eval_env(suite: str, task: str, motion_path: str, num_envs: int, seed: int, device: str):
    env_cfg = load_env_cfg(task, play=False)
    env_cfg.episode_length_s = 10.0  # 500 steps at 50 Hz
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed

    motion_cmd = env_cfg.commands['motion']
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.motion_file = motion_path
    motion_cmd.sampling_mode = 'uniform'

    if suite == 'nominal':
        env_cfg.observations['actor'].enable_corruption = False
        env_cfg.events.pop('push_robot', None)
        motion_cmd.pose_range = {}
        motion_cmd.velocity_range = {}
        motion_cmd.joint_position_range = (0.0, 0.0)
    elif suite == 'perturbed':
        env_cfg.observations['actor'].enable_corruption = True
    else:
        raise ValueError(f'unknown suite {suite!r}')

    return RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device))


def load_policy(algo: str, ckpt: str, env, task: str, config_path: str, device: str):
    inner = env.unwrapped
    actor_obs_dim = inner.single_observation_space.spaces['actor'].shape[0]
    critic_obs_dim = inner.single_observation_space.spaces['critic'].shape[0]
    act_dim = inner.single_action_space.shape[0]

    if algo == 'random':
        def policy(_obs_dict):
            return torch.empty(inner.num_envs, act_dim, device=device).uniform_(-1.0, 1.0)
        return policy

    if algo == 'sac':
        if ckpt.startswith('wandb://'):
            ckpt_path, rms_path = resolve_wandb_file(ckpt)
        else:
            ckpt_path = ckpt
            cand = ckpt.replace('.pt', '.rms.pt')
            rms_path = cand if os.path.exists(cand) else None

        cfg = load_config(config_path)
        cfg.device = device
        cfg.num_envs = 1
        SACAgent.infer_config_from_ckpt(ckpt_path, cfg, device)

        agent = SACAgent(actor_obs_dim, critic_obs_dim, act_dim, cfg)
        agent.load(ckpt_path)
        agent.actor.eval()

        actor_rms = EmpiricalNormalization(actor_obs_dim, torch.device(device))
        if rms_path and os.path.exists(rms_path):
            sd = torch.load(rms_path, map_location=device)
            actor_rms.load_state_dict(sd['actor_rms'])
        else:
            print(f'[WARN] no .rms.pt sidecar for {ckpt}')

        @torch.no_grad()
        def policy(obs_dict):
            return agent.select_action(actor_rms(obs_dict['actor']), deterministic=True)
        return policy

    if algo == 'ppo':
        ckpt_path = resolve_wandb_file(ckpt)[0] if ckpt.startswith('wandb://') else ckpt
        agent_cfg = load_rl_cfg(task)
        runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device=device)
        runner.load(str(ckpt_path), map_location=device)
        inf = runner.get_inference_policy(device=device)

        @torch.no_grad()
        def policy(obs_dict):
            return inf(obs_dict)
        return policy

    raise ValueError(f'unknown algo {algo!r}')


@torch.no_grad()
def evaluate_policy(env, policy, n_episodes: int, seed: int, device: str) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    inner = env.unwrapped
    num_envs = inner.num_envs
    act_dim = inner.single_action_space.shape[0]
    motion_cmd = inner.command_manager.get_term('motion')
    robot = inner.scene[motion_cmd.cfg.entity_name]

    obs_dict, _ = env.reset()
    motion_cmd._update_command()
    obs_dict = env.get_observations()

    cur_ret = torch.zeros(num_envs, device=device)
    cur_len = torch.zeros(num_envs, dtype=torch.long, device=device)
    cur_rpos = torch.zeros(num_envs, device=device)
    cur_rori = torch.zeros(num_envs, device=device)
    cur_bpos = torch.zeros(num_envs, device=device)
    cur_jpos = torch.zeros(num_envs, device=device)
    cur_arate = torch.zeros(num_envs, device=device)
    prev_act = torch.zeros(num_envs, act_dim, device=device)

    returns, lengths, completed = [], [], []
    rpos_ep, rori_ep, bpos_ep, jpos_ep, arate_ep = [], [], [], [], []

    while len(returns) < n_episodes:
        action = policy(obs_dict)
        arate_step = torch.norm(action - prev_act, dim=-1)
        prev_act = action

        # Measure tracking errors pre-step so auto-reset never contaminates done envs
        rpos_err = torch.norm(motion_cmd.anchor_pos_w - motion_cmd.robot_anchor_pos_w, dim=-1)
        rori_err = quat_error_magnitude(motion_cmd.anchor_quat_w, motion_cmd.robot_anchor_quat_w)
        bpos_err = torch.norm(motion_cmd.body_pos_relative_w - motion_cmd.robot_body_pos_w, dim=-1).mean(dim=-1)
        jpos_err = torch.abs(motion_cmd.joint_pos - robot.data.joint_pos).mean(dim=-1)

        obs_dict, rew, dones, extras = env.step(action)
        time_outs = extras.get('time_outs', torch.zeros_like(dones))

        cur_ret += rew
        cur_len += 1
        cur_rpos += rpos_err
        cur_rori += rori_err
        cur_bpos += bpos_err
        cur_jpos += jpos_err
        cur_arate += arate_step

        for i in dones.nonzero(as_tuple=True)[0].tolist():
            L = max(int(cur_len[i].item()), 1)
            returns.append(cur_ret[i].item())
            lengths.append(L)
            completed.append(float(bool(time_outs[i].item())))
            rpos_ep.append(cur_rpos[i].item() / L)
            rori_ep.append(cur_rori[i].item() / L)
            bpos_ep.append(cur_bpos[i].item() / L)
            jpos_ep.append(cur_jpos[i].item() / L)
            arate_ep.append(cur_arate[i].item() / L)

            cur_ret[i] = 0.0
            cur_len[i] = 0
            cur_rpos[i] = 0.0
            cur_rori[i] = 0.0
            cur_bpos[i] = 0.0
            cur_jpos[i] = 0.0
            cur_arate[i] = 0.0
            prev_act[i].zero_()
            if len(returns) >= n_episodes:
                break

    def stats(xs):
        arr = np.asarray(xs[:n_episodes], dtype=np.float64)
        return float(arr.mean()), float(arr.std())

    ret_m, ret_s = stats(returns)
    len_m, len_s = stats(lengths)
    return {
        'n_episodes': n_episodes,
        'return_mean': ret_m,
        'return_std': ret_s,
        'return_pct_max': 100.0 * ret_m / 50.0,
        'length_mean': len_m,
        'length_std': len_s,
        'completion_pct': 100.0 * float(np.mean(completed[:n_episodes])),
        'root_pos_err_m': float(np.mean(rpos_ep[:n_episodes])),
        'root_ori_err_rad': float(np.mean(rori_ep[:n_episodes])),
        'body_pos_err_m': float(np.mean(bpos_ep[:n_episodes])),
        'joint_pos_err_rad': float(np.mean(jpos_ep[:n_episodes])),
        'action_rate_l2': float(np.mean(arate_ep[:n_episodes])),
    }


def print_table(suite: str, rows: list[tuple[str, dict]]) -> None:
    print(f'\n### Suite: `{suite}`')
    print('| Model | Return (mean ± std) | % Max | Length (/500) | Complete % | Root Pos (m) | Root Ori (rad) | Body Pos (m) | Joint (rad) | Act Rate |')
    print('| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |')
    for name, m in rows:
        print(
            f"| {name} | {m['return_mean']:.2f} ± {m['return_std']:.2f} | {m['return_pct_max']:.1f}% | "
            f"{m['length_mean']:.1f} ± {m['length_std']:.1f} | {m['completion_pct']:.1f}% | "
            f"{m['root_pos_err_m']:.3f} | {m['root_ori_err_rad']:.3f} | "
            f"{m['body_pos_err_m']:.3f} | {m['joint_pos_err_rad']:.3f} | {m['action_rate_l2']:.2f} |"
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', nargs='+', required=True, help='name:algo:ckpt entries')
    p.add_argument('--suites', nargs='+', default=['nominal', 'perturbed'], choices=['nominal', 'perturbed'])
    p.add_argument('--task', default='Unitree-G1-Tracking')
    p.add_argument('--motion-path', default='data/lafan1_retargeted_npz/g1/walk1_subject1.npz')
    p.add_argument('--config', default='agents/configs/sac_config.yaml')
    p.add_argument('--num-envs', type=int, default=64)
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--json-out', default='outputs/eval/eval_suite.json')
    args = p.parse_args()

    specs = []
    for item in args.models:
        name, algo, ckpt = item.split(':', 2)
        specs.append((name, algo, ckpt))

    configure_torch_backends()
    all_results = {
        'config': {
            'task': args.task,
            'motion_path': args.motion_path,
            'episodes': args.episodes,
            'num_envs': args.num_envs,
            'seed': args.seed,
            'horizon_steps': 500,
        },
        'suites': {},
    }

    for suite in args.suites:
        env = build_eval_env(suite, args.task, args.motion_path, args.num_envs, args.seed, args.device)
        rows = []
        all_results['suites'][suite] = {}
        for name, algo, ckpt in specs:
            print(f'[EVAL] suite={suite} model={name} ({algo})')
            policy = load_policy(algo, ckpt, env, args.task, args.config, args.device)
            res = evaluate_policy(env, policy, args.episodes, args.seed, args.device)
            all_results['suites'][suite][name] = res
            rows.append((name, res))
        env.close()
        print_table(suite, rows)

    os.makedirs(os.path.dirname(args.json_out) or '.', exist_ok=True)
    with open(args.json_out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\n[DONE] wrote {args.json_out}')


if __name__ == '__main__':
    main()
