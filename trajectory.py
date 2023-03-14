from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self, Optional
from collections.abc import Sequence

import torch
from attrs import frozen
from torch import Tensor

if TYPE_CHECKING:
    # only needed for type annotations, can't import uncondionally due to import cycles
    from rl_player import TrainingInfo


@frozen
class Observation:
    observations: tuple[Tensor, ...]


@frozen
class Latent:
    latent: Tensor


@frozen(kw_only=True)
class TrajectoryState:
    """
    A list of TrajectoryStates represents a game trajectory a RLPlayer experienced.
    The data is intended to train a reinforcement learning agent and is thus
    agent-centric, e.g. recorded rewards are only valid for the agent which recorded the
    trajectory. In a multiplayer scenario therefore each agent has its own list of
    TrajectoryStates.
    The first TrajectoryState in a list is assumed to be the initial game state and thus
    should use inital tensors during network training.
    """

    representation: Observation | Latent
    belief: Optional[Tensor]
    current_player: int
    action: int
    target_policy: Sequence[float]
    mcts_value: float
    reward: float

    @classmethod
    def from_training_info(
        cls,
        info: TrainingInfo,
        *args: Any,
        target_policy: Optional[Sequence[float]] = None,
        current_player: int,
        action: int,
        reward: float,
    ) -> Self:
        return cls(
            representation=info.representation,
            belief=info.belief,
            current_player=current_player,
            action=action,
            target_policy=target_policy or info.target_policy,
            mcts_value=info.mcts_value,
            reward=reward,
        )


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
