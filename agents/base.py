from abc import ABC, abstractmethod
from typing import Any, Dict
import torch


class BaseAlgorithm(ABC):
    @abstractmethod
    def select_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass

    def get_metrics(self) -> Dict[str, Any]:
        return {}
