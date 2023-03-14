from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self, Optional
from collections.abc import Sequence

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
    is_observation: Tensor
    is_initial: Tensor
    is_data: Tensor

    observations: tuple[Tensor, ...]
    belief: Optional[Tensor]
    latent: Tensor
    current_player: Tensor
    action_onehot: Tensor
    target_policy: Tensor
    value_target: Tensor
    reward: Tensor
