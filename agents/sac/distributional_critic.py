import torch
import torch.nn as nn
import torch.nn.functional as F


class DistributionalQNetwork(nn.Module):
    def __init__(
        self,
        critic_obs_dim: int,
        act_dim: int,
        hidden_dims: list[int],
        num_atoms: int,
        v_min: float,
        v_max: float,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max

        layers = []
        in_dim = critic_obs_dim + act_dim
        for out_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, num_atoms))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))

    def projection(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        rewards: torch.Tensor,
        bootstrap: torch.Tensor,
        discount: torch.Tensor | float,
        q_support: torch.Tensor,
    ) -> torch.Tensor:
        """C51 Bellman projection onto fixed atom support."""
        device = q_support.device
        delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        batch_size = rewards.shape[0]
        discount_t = (
            discount if torch.is_tensor(discount)
            else torch.as_tensor(discount, device=device, dtype=q_support.dtype)
        )

        target_z = (rewards.unsqueeze(1) + bootstrap.unsqueeze(1) * discount_t * q_support).clamp(
            self.v_min, self.v_max
        )
        b = (target_z - self.v_min) / delta_z
        lower = torch.floor(b).long()
        upper = torch.ceil(b).long()

        # Separate lower/upper when landing exactly on an atom to avoid double-counting
        is_integer = upper == lower
        lower = torch.where(torch.logical_and(lower > 0, is_integer), lower - 1, lower)
        upper = torch.where(torch.logical_and(lower == 0, is_integer), upper + 1, upper)

        next_dist = F.softmax(self(state, action), dim=1)
        proj_dist = torch.zeros_like(next_dist)

        offset = (
            torch.linspace(0, (batch_size - 1) * self.num_atoms, batch_size, device=device)
            .unsqueeze(1)
            .expand(batch_size, self.num_atoms)
            .long()
        )
        max_index = proj_dist.numel() - 1
        lower_indices = torch.clamp((lower + offset).view(-1), 0, max_index)
        upper_indices = torch.clamp((upper + offset).view(-1), 0, max_index)

        proj_dist.view(-1).index_add_(0, lower_indices, (next_dist * (upper.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, upper_indices, (next_dist * (b - lower.float())).view(-1))
        return proj_dist


class DistributionalDoubleCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: list[int],
        num_atoms: int,
        v_min: float,
        v_max: float,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max

        self.Q1 = DistributionalQNetwork(obs_dim, act_dim, hidden_dims, num_atoms, v_min, v_max, use_layer_norm)
        self.Q2 = DistributionalQNetwork(obs_dim, act_dim, hidden_dims, num_atoms, v_min, v_max, use_layer_norm)
        self.register_buffer("q_support", torch.linspace(v_min, v_max, num_atoms))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.Q1(state, action), self.Q2(state, action)

    def get_value(self, probs: torch.Tensor) -> torch.Tensor:
        return (probs * self.q_support).sum(dim=-1, keepdim=True)

    def expected_Q(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        l1, l2 = self(state, action)
        return self.get_value(F.softmax(l1, dim=-1)), self.get_value(F.softmax(l2, dim=-1))

    def min_Q(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.expected_Q(state, action)
        return torch.min(q1, q2)

    def mean_Q(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.expected_Q(state, action)
        return 0.5 * (q1 + q2)

    def project_targets(
        self,
        next_state: torch.Tensor,
        next_action: torch.Tensor,
        rewards: torch.Tensor,
        bootstrap: torch.Tensor,
        discount: torch.Tensor | float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        p1 = self.Q1.projection(next_state, next_action, rewards, bootstrap, discount, self.q_support)
        p2 = self.Q2.projection(next_state, next_action, rewards, bootstrap, discount, self.q_support)
        return p1, p2
