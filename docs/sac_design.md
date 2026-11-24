# LocoMimic — SAC Design & Evolution Log

Notes documenting the evolution from vanilla SAC to GPU-parallel FastSAC-style and C51 distributional SAC on top of Unitree's `mjlab` framework for G1 whole-body motion imitation.

---

## 1. Baseline: Vanilla SAC
Starting point was a textbook single-env SAC implementation (see the `legacy-sac` branch):
- **Actor**: `tanh`-squashed Gaussian policy with squash-corrected `log_prob`.
- **Critic**: Double scalar Q (`Q1`, `Q2`), target bootstrap uses `min(Q1', Q2')`.
- **Entropy**: Autotuned `alpha` targeting `-act_dim`.
- **Buffer**: Flat ring buffer over individual transitions.
- **Optimizer**: Adam, 1 gradient step per env step.

---

## 2. Parallel GPU & FastSAC-Inspired Changes

Reference: FastTD3 / FastSAC (HoloSoma), Nauman et al. 2024 (BRO), Loshchilov & Hutter 2019 (AdamW).

1. **AdamW + weight decay**: `AdamW` (`betas=(0.9, 0.95)`, `fused=True` on CUDA) with `weight_decay=1e-3` on actor/critic and `0.0` on `log_alpha`.
2. **Vectorized target soft-updates inside the gradient loop**: `torch._foreach_mul_` + `torch._foreach_add_` after every gradient step (`tau=0.005`).
3. **LayerNorm + mean-of-Qs (`--fast-sac` / `sac_fast.yaml`)**: Critic LayerNorm after every linear layer paired with `0.5 * (Q1 + Q2)` aggregation.
4. **C51 Distributional Double Critic (`--distributional` / `sac_distributional.yaml`)**: 101-atom categorical value distribution (`v_min=-20`, `v_max=20`) with entropy-shifted Bellman projection and cross-entropy critic loss.
5. **TD3-style delayed policy updates (`policy_frequency`)**: Actor and alpha update every `N` critic updates (default `2`), with per-metric accumulator counts so logged averages stay unbiased.
6. **Critic gradient clipping (`critic_grad_clip=1.0`)**: Prevents critic norm spikes during high-UTD updates.
7. **Configurable entropy temperature (`use_autotune`, `alpha_lr`)**: Supports both autotuned alpha (with slower `alpha_lr=3e-5`) and fixed `alpha_init` for stability diagnostics.

---

## 3. Pipeline & Infrastructure

- **Per-env GPU replay buffer** (`agents/sac/replay_buffer.py`): Shaped `(buffer_size, num_envs, feat_dim)` with uniform `(step_idx, env_idx)` sampling.
- **Asymmetric actor-critic wrapper** (`rl/sac_env_wrapper.py`): Splits `actor` and privileged `critic` observations from `ManagerBasedRlEnv`.
- **Truncation-aware bootstrap + manual partial resets**: `auto_reset=False`; `done = terminated` (so `time_outs` still bootstrap), while resets fire on `terminated | time_outs`.
- **Online observation normalization (`EmpiricalNormalization`)**: Welford running mean/var per feature for actor and critic obs, updated on `step()` only (not `reset()`), saved alongside checkpoints as `<ckpt>.rms.pt`.
- **Self-describing checkpoints**: `agent.save()` records architecture config alongside weights; eval scripts also fall back to inferring distributional / LayerNorm flags from older checkpoints.

---

## 4. Key Bugs Fixed Along the Way

- **Missing obs normalization**: Raw mjlab observations span 100x scale differences; adding `EmpiricalNormalization` unlocked learning.
- **`tau=0.125` target instability**: With inner-loop target updates, `tau=0.125` caused Q oscillation and alpha runaway. Fixed by lowering to `tau=0.005`.
- **Partial reset indexing**: `env.reset(env_ids=finished)` returns full-batch tensors; indexed with `reset_obs[finished]`.
- **CPU `AdamW(fused=True)`**: Scoped `fused` to CUDA devices so CPU eval/tests run cleanly.
- **Task registration**: `import src.tasks` is required before `load_env_cfg('Unitree-G1-Tracking')`.
