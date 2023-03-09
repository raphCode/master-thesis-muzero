from typing import Optional
from collections.abc import Sequence

import torch
from attrs import frozen
from torch import Tensor


@frozen
class Observation:
    observations: tuple[Tensor, ...]


@frozen
class Latent:
    latent: Tensor


@frozen(kw_only=True)
class TrajectoryState:
    # the first TrajectoryState in a list is assumed to be with initial tensors
    representation: Observation | Latent
    belief: Optional[Tensor]
    current_player: int
    action: int
    target_policy: Sequence[float]
    mcts_value: float
    reward: float


@frozen(kw_only=True)
class TrainingData:
    is_observation: torch.Tensor
    is_data: torch.Tensor
    observation: tuple[torch.Tensor]
    latent_rep: torch.Tensor
    beliefs: torch.Tensor  # current or previous beliefs, depending on is_observation
    player_type: torch.Tensor
    action_onehot: torch.Tensor
    target_policy: torch.Tensor
    value_target: torch.Tensor
    reward: torch.Tensor
