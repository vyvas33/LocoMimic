import torch
import torch.nn as nn
from torch.distributions import Normal


class Actor(nn.Module):
    def __init__(
        self,
        actor_obs_dim,
        act_dim,
        hidden_dim: list[int],
        log_std_min: float = -20.0,
        log_std_max: float = 0.0,
        action_scale: float | list[float] | torch.Tensor = 1.0,
        init_log_std: float | None = None,
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        layers = []
        in_dim = actor_obs_dim
        for out_dim in hidden_dim:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim

        self.net = nn.Sequential(*layers)

        self.mean_head = nn.Linear(hidden_dim[-1], act_dim)
        self.log_std_head = nn.Linear(hidden_dim[-1], act_dim)

        if init_log_std is not None:
            nn.init.uniform_(self.mean_head.weight, -1e-3, 1e-3)
            nn.init.zeros_(self.mean_head.bias)
            nn.init.uniform_(self.log_std_head.weight, -1e-3, 1e-3)
            nn.init.constant_(self.log_std_head.bias, float(init_log_std))

        if isinstance(action_scale, str):
            raise ValueError(f"Resolve action_scale={action_scale!r} before building Actor")
        scale_t = torch.as_tensor(action_scale, dtype=torch.float32)
        if scale_t.ndim == 0:
            scale_t = scale_t.expand(act_dim).clone()
        self.register_buffer("action_scale", scale_t, persistent=False)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def deterministic(self, state):
        mean, _ = self.forward(state)
        return torch.tanh(mean) * self.action_scale

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        dist = Normal(mean, std)
        u = dist.rsample()

        y = torch.tanh(u)
        action = y * self.action_scale

        # Change-of-variables Jacobian for a = action_scale * tanh(u)
        log_prob = dist.log_prob(u) - torch.log(self.action_scale * (1.0 - y.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob
