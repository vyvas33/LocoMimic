from dataclasses import dataclass, field
import yaml


@dataclass
class SACConfig:
    motion_path: str
    device: str = "cuda:0"
    seed: int = 42
    wandb_project: str = "locomimic"
    actor_hidden_dim: list[int] = field(default_factory=lambda: [512, 512])
    critic_hidden_dim: list[int] = field(default_factory=lambda: [768, 768])
    lr: float = 3e-4
    weight_decay: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 4096
    buffer_size: int = 1024
    num_learning_iterations: int = 25000
    learning_starts: int = 10
    gradient_steps: int = 8
    policy_frequency: int = 1
    alpha_init: float = 0.01
    target_entropy_ratio: float = 0.5
    use_autotune: bool = True
    alpha_lr: float = 3e-5
    critic_grad_clip: float = 1.0
    log_std_min: float = -7.0
    log_std_max: float = 0.0
    action_scale: float | str | list[float] = 1.0
    init_log_std: float | None = None
    log_freq: int = 100
    ep_stats_window: int = 100
    ckpt_fractions: list[float] = field(default_factory=lambda: [0.25, 0.50, 0.75])
    use_layer_norm: bool = False
    use_mean_q: bool = False
    use_distributional: bool = False
    num_atoms: int = 101
    v_min: float = -20.0
    v_max: float = 20.0
    num_envs: int = 4096


def _auto_device(cfg: dict) -> dict:
    import torch
    if "device" not in cfg:
        cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg


def load_config(path: str) -> SACConfig:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return SACConfig(**_auto_device(cfg))
