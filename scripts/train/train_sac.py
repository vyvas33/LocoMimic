import argparse
import os
import time
from collections import deque
from datetime import datetime

import torch
import wandb

import mjlab.tasks  # noqa: F401
import src.tasks    # noqa: F401  registers Unitree-G1-Tracking

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.torch import configure_torch_backends

from agents.configs.config_loader import load_config
from agents.sac.sac_agent import SACAgent
from rl.sac_env_wrapper import SACVecEnvWrapper


def _accum_extras(extras, sums: dict, counts: dict) -> None:
    log = extras.get('log') if isinstance(extras, dict) else None
    if not log:
        return
    for k, v in log.items():
        if torch.is_tensor(v):
            if v.numel() != 1:
                continue
            val = v.detach().float()
        elif isinstance(v, (int, float)):
            val = float(v)
        else:
            continue
        if k in sums:
            sums[k] = sums[k] + val
            counts[k] += 1
        else:
            sums[k] = val.clone() if torch.is_tensor(val) else val
            counts[k] = 1


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--config', type=str, default='agents/configs/sac_config.yaml')
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--num_envs', type=int, default=4096)
    parser.add_argument('--task', type=str, default='Unitree-G1-Tracking')
    parser.add_argument('--fast-sac', action='store_true')
    parser.add_argument('--distributional', action='store_true')
    parser.add_argument('--autotune', dest='use_autotune', action='store_true', default=None)
    parser.add_argument('--no-autotune', dest='use_autotune', action='store_false')
    parser.add_argument('--alpha-init', type=float, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--iter', type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    config.num_envs = args.num_envs

    if args.iter is not None:
        config.num_learning_iterations = args.iter
    if args.seed is not None:
        config.seed = args.seed
    if args.fast_sac:
        config.use_layer_norm = True
        config.use_mean_q = True
    if args.distributional:
        config.use_distributional = True
    if args.use_autotune is not None:
        config.use_autotune = args.use_autotune
    if args.alpha_init is not None:
        config.alpha_init = args.alpha_init

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = args.name or f'sac_{timestamp}'
    run_model_dir = f'models/{run_name}'
    run_log_dir = f'logs/{run_name}'
    os.makedirs(run_model_dir, exist_ok=True)
    os.makedirs(run_log_dir, exist_ok=True)

    wandb.init(project=config.wandb_project, name=run_name, config=config.__dict__)

    configure_torch_backends()
    env_cfg = load_env_cfg(args.task)

    motion_cmd = env_cfg.commands['motion']
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.motion_file = config.motion_path
    motion_cmd.sampling_mode = 'uniform'

    env_cfg.auto_reset = False
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = config.seed

    env = SACVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=config.device))
    agent = SACAgent(env.actor_obs_dim, env.critic_obs_dim, env.act_dim, config)

    if args.load:
        agent.load(args.load)
        rms_path = args.load.replace('.pt', '.rms.pt')
        if os.path.exists(rms_path):
            env.load_state_dict(torch.load(rms_path, map_location=config.device))
            print(f'Loaded obs normalizer: {rms_path}')
        print(f'Loaded checkpoint: {args.load}')

    print(f'Run: {run_name} | Device: {config.device} | Envs: {args.num_envs} | Iters: {config.num_learning_iterations}')

    actor_obs, critic_obs, _ = env.reset()
    episode_returns = torch.zeros(args.num_envs, device=config.device)
    episode_steps = torch.zeros(args.num_envs, device=config.device)
    episode_num = 0

    recent_returns = deque(maxlen=config.ep_stats_window)
    recent_lengths = deque(maxlen=config.ep_stats_window)

    reward_sum = torch.zeros((), device=config.device)
    reward_sqsum = torch.zeros((), device=config.device)
    reward_count = 0
    term_sum = torch.zeros((), device=config.device)
    timeout_sum = torch.zeros((), device=config.device)
    extras_sums: dict = {}
    extras_counts: dict[str, int] = {}
    last_log_time = time.time()

    ckpt_steps = {int(config.num_learning_iterations * f) for f in config.ckpt_fractions}

    for step in range(config.num_learning_iterations):
        if step < config.learning_starts:
            action = torch.rand(args.num_envs, env.act_dim, device=config.device) * 2 - 1
        else:
            action = agent.select_action(actor_obs)

        next_actor_obs, next_critic_obs, reward, terminated, time_outs, step_extras = env.step(action)
        _accum_extras(step_extras, extras_sums, extras_counts)

        # Store terminated only as done so timeouts still bootstrap
        agent.collect(actor_obs, critic_obs, action, reward, next_actor_obs, next_critic_obs, terminated)

        episode_returns += reward
        episode_steps += 1
        reward_sum += reward.sum()
        reward_sqsum += (reward * reward).sum()
        reward_count += reward.numel()
        term_sum += terminated.sum()
        timeout_sum += time_outs.sum()

        done_any = terminated | time_outs
        if done_any.any():
            finished = done_any.nonzero(as_tuple=False).squeeze(-1)
            returns_cpu = episode_returns[finished].cpu().tolist()
            lengths_cpu = episode_steps[finished].cpu().tolist()
            recent_returns.extend(returns_cpu)
            recent_lengths.extend(lengths_cpu)
            episode_num += len(returns_cpu)

            episode_returns[finished] = 0.0
            episode_steps[finished] = 0.0

            reset_actor, reset_critic, reset_extras = env.reset(env_ids=finished)
            _accum_extras(reset_extras, extras_sums, extras_counts)
            next_actor_obs[finished] = reset_actor[finished]
            next_critic_obs[finished] = reset_critic[finished]

        if step >= config.learning_starts:
            agent.update()

        actor_obs = next_actor_obs
        critic_obs = next_critic_obs

        if step % config.log_freq == 0 and step > 0:
            now = time.time()
            dt = max(now - last_log_time, 1e-6)
            last_log_time = now

            log_data = {
                'env_step': step,
                'env/transitions': step * args.num_envs,
                'buffer/size': len(agent.buffer),
                'perf/iters_per_sec': config.log_freq / dt,
                'perf/env_steps_per_sec': config.log_freq * args.num_envs / dt,
            }

            if reward_count > 0:
                r_mean = (reward_sum / reward_count).item()
                r_var = (reward_sqsum / reward_count).item() - r_mean * r_mean
                log_data['env/reward_mean'] = r_mean
                log_data['env/reward_std'] = max(r_var, 0.0) ** 0.5
                reward_sum.zero_()
                reward_sqsum.zero_()
                reward_count = 0

            n_term = term_sum.item()
            n_timeout = timeout_sum.item()
            n_done = n_term + n_timeout
            if n_done > 0:
                log_data['done/terminated_rate'] = n_term / n_done
                log_data['done/timeout_rate'] = n_timeout / n_done
                term_sum.zero_()
                timeout_sum.zero_()

            if extras_counts:
                for k, total in extras_sums.items():
                    val = total.item() if torch.is_tensor(total) else total
                    log_data[f'mjlab/{k}'] = val / extras_counts[k]
                extras_sums.clear()
                extras_counts.clear()

            if recent_returns:
                log_data['episode/avg_return'] = sum(recent_returns) / len(recent_returns)
                log_data['episode/avg_steps'] = sum(recent_lengths) / len(recent_lengths)
                log_data['episode/num'] = episode_num

            log_data.update(agent.pop_metrics())
            log_data['policy/target_entropy'] = agent.target_entropy
            wandb.log(log_data, step=step)

            avg_r = log_data.get('episode/avg_return', float('nan'))
            avg_l = log_data.get('episode/avg_steps', float('nan'))
            qf = log_data.get('loss/critic', float('nan'))
            al = log_data.get('alpha/value', float('nan'))
            fps = log_data['perf/env_steps_per_sec']
            print(f'Step {step:7d} | Ep {episode_num:5d} | R={avg_r:7.2f} L={avg_l:5.1f} | qf={qf:.3f} α={al:.4f} | {fps:.0f} steps/s')

        if step in ckpt_steps:
            path = f'{run_model_dir}/ckpt_step_{step}.pt'
            rms_path = path.replace('.pt', '.rms.pt')
            agent.save(path)
            torch.save(env.state_dict(), rms_path)
            wandb.save(path, base_path=run_model_dir, policy='now')
            wandb.save(rms_path, base_path=run_model_dir, policy='now')
            print(f'Checkpoint saved: {path}')

    final_path = f'{run_model_dir}/final.pt'
    final_rms = final_path.replace('.pt', '.rms.pt')
    agent.save(final_path)
    torch.save(env.state_dict(), final_rms)
    wandb.save(final_path, base_path=run_model_dir, policy='now')
    wandb.save(final_rms, base_path=run_model_dir, policy='now')
    print(f'Final model saved: {final_path}')

    env.close()
    wandb.finish()


if __name__ == '__main__':
    train()
