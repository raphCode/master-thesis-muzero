from __future__ import annotations

import operator
import functools
from typing import TYPE_CHECKING, Any, Iterable, Optional, TypeAlias, cast

import attrs
import torch
from attrs import frozen
from torch import Tensor

from mcts import TurnStatus
from util import TensorCache, ndarr_f64
from config import C

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
    turn_status: int
    action: int
    target_policy: ndarr_f64
    mcts_value: float
    reward: float

    @classmethod
    def from_training_info(
        cls,
        info: TrainingInfo,
        *args: Any,
        target_policy: Optional[ndarr_f64] = None,
        turn_status: int,
        action: int,
        reward: float,
    ) -> TrajectoryState:
        if target_policy is None:
            target_policy = info.target_policy
        return cls(
            representation=info.representation,
            turn_status=turn_status,
            action=action,
            target_policy=target_policy,
            mcts_value=info.mcts_value,
            reward=reward,
        )


@frozen(kw_only=True)
class TrainingData:
    """
    Similar to TrajectoryState, but in a form more convenient for network training.
    A list of TrainingData is a complete training batch, it stores multiple continous
    sections of game trajectories stacked in the batch dimension.
    The list length is between min_trajectory_length and max_trajectory_length, and the
    stacked height inside each tensor of a single TrainingData is batch_size.

                    timestep / list index
    list[TrainingData]    0 1 2 3 4
                         |         |
                    trajectory move number
    trajectory a  0 1 2 3|4 5 6 7 8|9 ...      ^
    trajectory b         |0 1 2 3 4|5 6 7 8 9  | batch_size
    trajectory c      0 1|2 3 4 D D|           v
                          <------->
                 [min-max]_trajectory_length

    D = dummy TrainingData to pad trajectories beyond their end up to batch trajectory
    length

    Due to the nature of batching, each timestep is processed at once, but may contain
    different types of trajectory states (initial state, observation / own move, other
    player move).
    To select the correct behavior during training for each trajectory, this requires the
    use of boolean masks.
    """

    # masks:
    is_observation: Tensor
    is_initial: Tensor
    is_data: Tensor

    observations: tuple[Tensor, ...]
    latent: Tensor
    turn_status: Tensor
    action_onehot: Tensor
    target_policy: Tensor
    value_target: Tensor
    reward: Tensor

    @classmethod  # type: ignore [misc]
    @property
    @cast("functools._lru_cache_wrapper[TrainingData]", functools.cache)
    def dummy(cls) -> TrainingData:
        """
        Dummy data for padding trajectories ending early inside the batch.
        Designed to use as little memory as possible:
        - tensor instances with identical data reused internally
        - returned TrainingData instance cached for future calls
        """
        cache = TensorCache()
        return cls(
            is_observation=cache.tensor(False),
            is_initial=cache.tensor(False),
            is_data=cache.tensor(False),
            observations=tuple(map(torch.zeros, C.game.instance.observation_shapes)),
            latent=cache.zeros(C.networks.latent_shape),
            turn_status=cache.tensor(0, dtype=torch.long),  # index tensor needs long
            action_onehot=cache.zeros(C.game.instance.max_num_actions, dtype=torch.long),
            target_policy=cache.zeros(C.game.instance.max_num_actions),
            value_target=cache.tensor(0),
            reward=cache.tensor(0.0),
        )

    @classmethod
    def from_trajectory_state(
        cls,
        ts: TrajectoryState,
        value_target: float,
        is_initial: bool,
        is_terminal: bool,
        cache: Optional[TensorCache] = None,
    ) -> TrainingData:
        if cache is None:
            cache = TensorCache()
        is_obs = isinstance(ts.representation, Observation)
        return cls(
            is_observation=cache.tensor(is_obs),
            is_initial=cache.tensor(bool(is_initial)),
            is_data=cache.tensor(True),
            observations=cast(
                TrainingData | Observation,
                ts.representation if is_obs else cls.dummy,
            ).observations,
            latent=cast(
                TrainingData | Latent,
                ts.representation if not is_obs else cls.dummy,
            ).latent,
            turn_status=cache.tensor(
                TurnStatus.TERMINAL_STATE.target_index if is_terminal else ts.turn_status,
                dtype=torch.long,
            ),
            action_onehot=cache.onehot(
                ts.action,
                C.game.instance.max_num_actions,
            ),
            target_policy=torch.tensor(ts.target_policy),
            value_target=torch.tensor(float(value_target)),
            reward=cache.tensor(float(ts.reward)),
        )

    @classmethod
    def stack_batch(cls, instances: Iterable[TrainingData]) -> TrainingData:
        """
        Stacks the tensors of all instances in the first (batch) dimension.
        """
        FieldTypesUnion: TypeAlias = Optional[Tensor] | tuple[Tensor, ...]
        stacked_data: FieldTypesUnion
        stacked_fields = dict[str, FieldTypesUnion]()

        field_names = [f.name for f in attrs.fields(cls)]
        get_fields = operator.attrgetter(*field_names)
        field_data = zip(*map(get_fields, instances))
        for name, data in zip(field_names, field_data):
            # TODO: save memory by setting latent,beliefs = None for all steps expect first?
            if name == "observations":
                stacked_data = tuple(map(torch.stack, zip(*data)))
            else:
                if not (name.startswith("is_") or name in ("turn_status", "belief")):
                    data = torch.atleast_1d(data)  # type: ignore [no-untyped-call]
                    # only keep this data 0D:
                    # - is_*: boolean masks, processed specially in the training
                    # - turn_status: cross entropy loss takes class indices directly
                    # - belief: may be empty
                stacked_data = torch.stack(data)
            stacked_fields[name] = stacked_data
        return cls(**stacked_fields)  # type: ignore [arg-type]
