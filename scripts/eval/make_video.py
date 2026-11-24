"""Side-by-side reference (left) vs. policy (right) videos (mp4 + gif).

usage:
  mjpython -m scripts.eval.make_video --algo random
  mjpython -m scripts.eval.make_video --algo sac      --ckpt path/to/ckpt.pt
  mjpython -m scripts.eval.make_video --algo fast_sac --ckpt path/to/ckpt.pt
  mjpython -m scripts.eval.make_video --algo ppo      --ckpt path/to/ckpt.pt
"""

import argparse
import copy
import os
import time
from pathlib import Path

import imageio
import mujoco
import numpy as np
import torch
from PIL import Image

import mjlab.tasks  # noqa: F401
import src.tasks    # noqa: F401  registers Unitree-G1-Tracking

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.torch import configure_torch_backends


class RobotRenderer:
    def __init__(self, model: mujoco.MjModel, width: int, height: int,
                 track_body_id: int, distance: float, elevation: float, azimuth: float):
        self._model = copy.deepcopy(model)
        self._model.vis.global_.offwidth = width
        self._model.vis.global_.offheight = height
        self._model.light_castshadow[:] = False
        self._model.mat_reflectance[:] = 0.0
        self._data = mujoco.MjData(self._model)
        self._renderer = mujoco.Renderer(self._model, height=height, width=width)

        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self._model, cam)
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING.value
        cam.trackbodyid = track_body_id
        cam.fixedcamid = -1
        cam.distance = distance
        cam.elevation = elevation
        cam.azimuth = azimuth
        self._cam = cam

    def render(self, qpos: np.ndarray) -> np.ndarray:
        self._data.qpos[:] = qpos
        self._data.qvel[:] = 0.0
        mujoco.mj_forward(self._model, self._data)
        self._renderer.update_scene(self._data, camera=self._cam)
        return self._renderer.render()

    def close(self):
        self._renderer.close()


def _make_random_policy(env):
    inner = env.unwrapped
    act_dim = inner.single_action_space.shape[0]

    def policy(_obs):
        return torch.empty(inner.num_envs, act_dim, device=inner.device).uniform_(-1.0, 1.0)
    return policy


def _make_sac_policy(env, args, fast_sac: bool):
    from agents.configs.config_loader import load_config
    from agents.sac.sac_agent import SACAgent
    from rl.sac_env_wrapper import EmpiricalNormalization

    cfg = load_config(args.config or 'agents/configs/sac_config.yaml')
    cfg.device = args.device
    cfg.num_envs = 1
    if fast_sac:
        cfg.use_layer_norm = True
        cfg.use_mean_q = True
    SACAgent.infer_config_from_ckpt(args.ckpt, cfg, args.device)

    inner = env.unwrapped
    actor_obs_dim = inner.single_observation_space.spaces['actor'].shape[0]
    critic_obs_dim = inner.single_observation_space.spaces['critic'].shape[0]
    act_dim = inner.single_action_space.shape[0]

    agent = SACAgent(actor_obs_dim, critic_obs_dim, act_dim, cfg)
    agent.load(args.ckpt)

    rms = EmpiricalNormalization(actor_obs_dim, torch.device(args.device))
    rms_path = args.ckpt.replace('.pt', '.rms.pt')
    if os.path.exists(rms_path):
        sd = torch.load(rms_path, map_location=args.device)
        rms.load_state_dict(sd['actor_rms'])
    else:
        print(f'[WARN] no .rms.pt sidecar at {rms_path}')

    @torch.no_grad()
    def policy(obs_dict):
        return agent.select_action(rms(obs_dict['actor']), deterministic=True)
    return policy


def _resolve_wandb_ckpt(uri: str, cache_dir: str = 'outputs/wandb_ckpts') -> str:
    """Download/cache wandb://entity/project/run_name[:filename] locally."""
    import re
    import wandb

    rest = uri.removeprefix('wandb://')
    run_path, filename = rest.split(':', 1) if ':' in rest else (rest, None)
    parts = run_path.split('/')
    if len(parts) != 3:
        raise ValueError(f'expected wandb://entity/project/run_name[:file], got {uri!r}')
    entity, project, run_name = parts

    api = wandb.Api(timeout=60)
    runs = list(api.runs(f'{entity}/{project}', filters={'display_name': run_name}))
    if not runs:
        raise ValueError(f'no run named {run_name!r} in {entity}/{project}')
    run = runs[0]

    if filename is None:
        ckpts = [f for f in run.files() if re.fullmatch(r'model_\d+\.pt', f.name)]
        if not ckpts:
            raise ValueError(f'run {run_name!r} has no model_*.pt files')
        ckpts.sort(key=lambda f: int(re.search(r'\d+', f.name).group()))
        target = ckpts[-1]
        filename = target.name
    else:
        target = next((f for f in run.files() if f.name == filename), None)
        if target is None:
            raise ValueError(f'file {filename!r} not in run {run_name!r}')

    local_dir = os.path.join(cache_dir, run_name)
    local_path = os.path.join(local_dir, filename)
    if not os.path.exists(local_path):
        os.makedirs(local_dir, exist_ok=True)
        target.download(root=local_dir, replace=False)
    return local_path


def _make_ppo_policy(env, args):
    from dataclasses import asdict
    from mjlab.rl import MjlabOnPolicyRunner
    from mjlab.tasks.registry import load_rl_cfg, load_runner_cls

    ckpt_path = _resolve_wandb_ckpt(args.ckpt) if args.ckpt.startswith('wandb://') else args.ckpt
    rl_cfg = load_rl_cfg(args.task)
    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(rl_cfg), device=args.device)
    runner.load(ckpt_path, load_cfg={'actor': True}, strict=True, map_location=args.device)
    inf = runner.get_inference_policy(device=args.device)

    @torch.no_grad()
    def policy(obs_dict):
        return inf(obs_dict)
    return policy


def make_policy(algo: str, env, args):
    if algo == 'random':   return _make_random_policy(env)
    if algo == 'sac':      return _make_sac_policy(env, args, fast_sac=False)
    if algo == 'fast_sac': return _make_sac_policy(env, args, fast_sac=True)
    if algo == 'ppo':      return _make_ppo_policy(env, args)
    raise ValueError(f'unknown algo {algo!r}')


def build_env(args):
    env_cfg = load_env_cfg(args.task, play=True)
    motion_cmd_cfg = env_cfg.commands['motion']
    assert isinstance(motion_cmd_cfg, MotionCommandCfg)
    if args.motion_path:
        motion_cmd_cfg.motion_file = args.motion_path
    motion_cmd_cfg.sampling_mode = 'uniform'
    motion_cmd_cfg.pose_range = {}
    motion_cmd_cfg.velocity_range = {}
    motion_cmd_cfg.joint_position_range = (0.0, 0.0)

    env_cfg.scene.num_envs = 1
    env_cfg.seed = args.seed
    env_cfg.terminations = {}

    return RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=args.device)), motion_cmd_cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--algo', required=True, choices=['random', 'sac', 'fast_sac', 'ppo'])
    p.add_argument('--ckpt', type=str, default=None)
    p.add_argument('--config', type=str, default=None)
    p.add_argument('--motion-path', type=str, default='data/lafan1_retargeted_npz/g1/walk1_subject1.npz')
    p.add_argument('--task', type=str, default='Unitree-G1-Tracking')
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--steps', type=int, default=250)
    p.add_argument('--episodes', type=int, default=3)
    p.add_argument('--init-frames', type=str, default='1000,4000,600')
    p.add_argument('--width', type=int, default=480)
    p.add_argument('--height', type=int, default=360)
    p.add_argument('--gif-fps', type=int, default=25)
    p.add_argument('--gif-width', type=int, default=480)
    p.add_argument('--cam-distance', type=float, default=2.5)
    p.add_argument('--cam-elevation', type=float, default=-15.0)
    p.add_argument('--cam-azimuth', type=float, default=120.0)
    p.add_argument('--out-dir', type=str, default='outputs/videos')
    args = p.parse_args()

    if args.algo != 'random' and args.ckpt is None:
        p.error(f'--ckpt is required for --algo {args.algo}')

    init_frames: list[int] | None = None
    if args.init_frames:
        init_frames = [int(x) for x in args.init_frames.split(',') if x.strip()]
        if len(init_frames) != args.episodes:
            p.error(f'--init-frames has {len(init_frames)} entries but --episodes is {args.episodes}')

    configure_torch_backends()
    env, _ = build_env(args)
    inner = env.unwrapped
    motion_cmd = inner.command_manager.get_term('motion')

    next_frame_idx = [0]
    if init_frames is not None:
        total = int(motion_cmd.motion.time_step_total)
        for f in init_frames:
            if not (0 <= f < total):
                p.error(f'--init-frames contains {f} but motion has {total} frames')

        def _fixed_sampling(env_ids, _mcmd=motion_cmd):
            _mcmd.time_steps[env_ids] = init_frames[next_frame_idx[0]]

        motion_cmd._uniform_sampling = _fixed_sampling
        motion_cmd._adaptive_sampling = _fixed_sampling

    robot = inner.scene[motion_cmd.cfg.entity_name]
    anchor_body_ids, _ = robot.find_bodies(motion_cmd.cfg.anchor_body_name)
    track_body_id = robot.indexing.bodies[anchor_body_ids[0]].id

    free_q_adr = robot.indexing.free_joint_q_adr.cpu().numpy()
    joint_q_adr = robot.indexing.joint_q_adr.cpu().numpy()

    common = dict(
        model=inner.sim.mj_model,
        width=args.width, height=args.height,
        track_body_id=track_body_id,
        distance=args.cam_distance, elevation=args.cam_elevation, azimuth=args.cam_azimuth,
    )
    ref_renderer = RobotRenderer(**common)
    pol_renderer = RobotRenderer(**common)
    policy = make_policy(args.algo, env, args)

    nq = inner.sim.mj_model.nq
    frames: list[np.ndarray] = []
    fps = float(1.0 / inner.step_dt)

    for ep in range(args.episodes):
        next_frame_idx[0] = ep
        obs_dict, _ = env.reset()

        for _ in range(args.steps):
            actual_qpos = inner.sim.data.qpos[0].cpu().numpy()
            ref_qpos = np.zeros(nq)
            ref_qpos[free_q_adr[0:3]] = (
                motion_cmd.body_pos_w[0, 0].cpu().numpy() - inner.scene.env_origins[0].cpu().numpy()
            )
            ref_qpos[free_q_adr[3:7]] = motion_cmd.body_quat_w[0, 0].cpu().numpy()
            ref_qpos[joint_q_adr] = motion_cmd.joint_pos[0].cpu().numpy()

            frames.append(np.concatenate([ref_renderer.render(ref_qpos), pol_renderer.render(actual_qpos)], axis=1))
            with torch.no_grad():
                action = policy(obs_dict)
            obs_dict, _, _, _ = env.step(action)

    env.close()
    ref_renderer.close()
    pol_renderer.close()

    stamp = time.strftime('%Y%m%d-%H%M%S')
    out_dir = Path(args.out_dir) / f'{args.algo}_{stamp}'
    out_dir.mkdir(parents=True, exist_ok=True)

    mp4_path = out_dir / f'{args.algo}.mp4'
    gif_path = out_dir / f'{args.algo}.gif'
    imageio.mimwrite(mp4_path, frames, fps=fps, codec='libx264', quality=8, macro_block_size=1)

    h, w = frames[0].shape[:2]
    if args.gif_width < w:
        new_size = (args.gif_width, int(h * (args.gif_width / w)))
        gif_frames = [np.array(Image.fromarray(f).resize(new_size, Image.BILINEAR)) for f in frames]
    else:
        gif_frames = frames

    stride = max(1, int(round(fps / args.gif_fps)))
    imageio.mimwrite(gif_path, gif_frames[::stride], fps=args.gif_fps, loop=0)
    print(f'[DONE] {out_dir}')


if __name__ == '__main__':
    main()
