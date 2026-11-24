"""Plot mean episode return and length curves for PPO / SAC variants from wandb.

usage:
  python -m scripts.eval.plot_training
  python -m scripts.eval.plot_training --mod1 --smooth 1_000_000 --max-steps 5e7
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb
from matplotlib.ticker import FuncFormatter


ALGOS = {
    'ppo': {
        'return_keys': ['Train/mean_reward'],
        'length_keys': ['Train/mean_episode_length'],
        'x_key': '_step',
        'x_kind': 'iteration',
        'color': '#1f77b4',
        'label': 'PPO',
    },
    'sac': {
        'return_keys': ['episode/avg_return'],
        'length_keys': ['episode/avg_steps'],
        'x_key': 'env/transitions',
        'x_kind': 'env_step',
        'color': '#d62728',
        'label': 'SAC',
    },
    'fastsac': {
        'return_keys': ['episode/avg_return'],
        'length_keys': ['episode/avg_steps'],
        'x_key': 'env/transitions',
        'x_kind': 'env_step',
        'color': '#9467bd',
        'label': 'FastSAC',
    },
}


def find_run(api: wandb.Api, entity: str, project: str, name: str):
    matches = list(api.runs(f'{entity}/{project}', filters={'display_name': name}))
    if matches:
        return matches[0]
    all_runs = list(api.runs(f'{entity}/{project}'))
    subs = [r for r in all_runs if name in r.name]
    if not subs:
        available = ', '.join(sorted({r.name for r in all_runs}))
        raise ValueError(f'no run named {name!r} in {entity}/{project}. available: {available}')
    return subs[0]


def fetch_history(run, metric_keys: list[str], x_key: str, samples: int, full: bool):
    summary_keys = set(run.summary.keys()) | set(run.history(samples=1, pandas=False)[0].keys())
    chosen = next((k for k in metric_keys if k in summary_keys), None)
    if chosen is None:
        raise ValueError(f'run {run.name!r}: none of {metric_keys} found')

    rows = (
        run.scan_history(keys=[chosen, x_key])
        if full
        else run.history(keys=[chosen, x_key], samples=samples, pandas=False)
    )
    xs, ys = [], []
    for row in rows:
        x, y = row.get(x_key), row.get(chosen)
        if x is not None and y is not None:
            xs.append(x)
            ys.append(y)
    if not xs:
        raise ValueError(f'run {run.name!r}: empty history for ({chosen}, {x_key})')
    order = np.argsort(xs)
    return np.array(xs, dtype=np.float64)[order], np.array(ys, dtype=np.float64)[order], chosen


def x_multiplier(run, x_kind: str) -> float:
    if x_kind == 'env_step':
        return 1.0
    if x_kind == 'iteration':
        cfg = run.config

        def get(d, *path):
            cur = d
            for p in path:
                if isinstance(cur, dict) and p in cur:
                    cur = cur[p]
                else:
                    return None
            return cur

        nse = (get(cfg, 'num_steps_per_env') or get(cfg, 'train_cfg', 'num_steps_per_env')
               or get(cfg, 'algorithm', 'num_steps_per_env') or get(cfg, 'runner', 'num_steps_per_env'))
        ne = (get(cfg, 'num_envs') or get(cfg, 'env_cfg', 'scene', 'num_envs')
              or get(cfg, 'env', 'num_envs') or get(cfg, 'scene', 'num_envs'))
        if nse is None or ne is None:
            return 1.0
        return float(nse) * float(ne)
    raise ValueError(f'unknown x_kind {x_kind!r}')


def smooth_xy(x: np.ndarray, y: np.ndarray, window: float) -> np.ndarray:
    if window <= 0:
        return y.copy()
    half = window / 2.0
    out = np.empty_like(y)
    j_lo = j_hi = 0
    n = len(x)
    for i in range(n):
        while j_lo < n and x[j_lo] < x[i] - half:
            j_lo += 1
        while j_hi < n and x[j_hi] <= x[i] + half:
            j_hi += 1
        out[i] = y[j_lo:j_hi].mean()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--entity', default='locomimic')
    p.add_argument('--project', default='locomimic')
    p.add_argument('--ppo-run', default='PPO_4096')
    p.add_argument('--sac-run', default='sac1_v5_distQ')
    p.add_argument('--fastsac-run', default='sac1_fast_v1')
    p.add_argument('--mod1', action='store_true', help='compare SAC vs FastSAC instead of PPO vs SAC')
    p.add_argument('--smooth', type=float, default=5e5)
    p.add_argument('--smooth-ppo', type=float, default=None)
    p.add_argument('--smooth-sac', type=float, default=None)
    p.add_argument('--smooth-fastsac', type=float, default=None)
    p.add_argument('--max-steps', type=float, default=None)
    p.add_argument('--samples', type=int, default=1500)
    p.add_argument('--full', action='store_true')
    p.add_argument('--log-x', action='store_true')
    p.add_argument('--x-units', choices=['raw', 'M', 'B'], default='M')
    p.add_argument('--out', default='outputs/plots/training.png')
    p.add_argument('--random-baseline', type=str, default=None)
    p.add_argument('--width', type=float, default=12)
    p.add_argument('--height', type=float, default=4.5)
    p.add_argument('--dpi', type=int, default=160)
    args = p.parse_args()

    api = wandb.Api(timeout=60)
    run_specs = (
        {'sac': args.sac_run, 'fastsac': args.fastsac_run}
        if args.mod1
        else {'ppo': args.ppo_run, 'sac': args.sac_run}
    )
    smooth_overrides = {'ppo': args.smooth_ppo, 'sac': args.smooth_sac, 'fastsac': args.smooth_fastsac}
    runs = {algo: find_run(api, args.entity, args.project, name) for algo, name in run_specs.items()}

    def _pull(algo):
        spec, run = ALGOS[algo], runs[algo]
        r = fetch_history(run, spec['return_keys'], spec['x_key'], args.samples, args.full)
        l = fetch_history(run, spec['length_keys'], spec['x_key'], args.samples, args.full)
        return algo, r, l

    series = {}
    with ThreadPoolExecutor(max_workers=len(run_specs) * 2) as pool:
        for algo, (x_r, y_r, _), (x_l, y_l, _) in pool.map(_pull, run_specs):
            mult = x_multiplier(runs[algo], ALGOS[algo]['x_kind'])
            x_r, x_l = x_r * mult, x_l * mult
            if args.max_steps is not None:
                m_r, m_l = x_r <= args.max_steps, x_l <= args.max_steps
                x_r, y_r, x_l, y_l = x_r[m_r], y_r[m_r], x_l[m_l], y_l[m_l]
            series[algo] = {'return': (x_r, y_r), 'length': (x_l, y_l)}

    fig, (ax_r, ax_l) = plt.subplots(1, 2, figsize=(args.width, args.height), dpi=args.dpi)
    unit_div = {'raw': 1.0, 'M': 1e6, 'B': 1e9}[args.x_units]
    unit_suffix = {'raw': '', 'M': ' (M)', 'B': ' (B)'}[args.x_units]

    for algo, data in series.items():
        spec = ALGOS[algo]
        smooth_w = smooth_overrides.get(algo) or args.smooth
        for ax, key, ylabel in [(ax_r, 'return', 'mean episode return'), (ax_l, 'length', 'mean episode length (steps)')]:
            x, y = data[key]
            ax.plot(x, y, color=spec['color'], alpha=0.18, linewidth=0.8)
            ax.plot(x, smooth_xy(x, y, smooth_w), color=spec['color'], linewidth=2.0, label=spec['label'])
            ax.set_ylabel(ylabel)
            ax.set_xlabel(f'environment steps{unit_suffix}')
            ax.grid(True, alpha=0.25)

    if args.random_baseline:
        with open(args.random_baseline) as f:
            rb = json.load(f)
        ax_r.axhline(rb['return_mean'], color='gray', linestyle='--', linewidth=1.5, label='Random')
        ax_l.axhline(rb['length_mean'], color='gray', linestyle='--', linewidth=1.5, label='Random')

    for ax in (ax_r, ax_l):
        if len(series) > 1 or args.random_baseline:
            ax.legend(loc='best', frameon=True)
        if args.log_x:
            ax.set_xscale('log')
        elif args.x_units == 'raw':
            ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        else:
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _, d=unit_div: f'{v/d:g}'))

    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    print(f'[DONE] wrote {out_path}')


if __name__ == '__main__':
    main()
