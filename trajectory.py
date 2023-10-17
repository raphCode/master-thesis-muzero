from __future__ import annotations

import operator
import functools
from typing import TYPE_CHECKING, Iterable, Optional, TypeAlias, cast

import attrs
import torch
from attrs import frozen
from torch import Tensor

from mcts import TurnStatus
from util import TensorCache
from config import C

if TYPE_CHECKING:
    # only needed for type annotations, can't import uncondionally due to import cycles
    from util import ndarr_f32


@frozen(kw_only=True)
class TrajectoryState:
    """
    A list of TrajectoryStates represents a game trajectory.
    Each TrajectoryState corresponds to a game state.
    Prediction targets for the dynamics network actually refer to the next state,
    specifically the reward and turn status.
    """

    observations: Optional[tuple[Tensor, ...]]
    turn_status: int
    action: int
    target_policy: ndarr_f32
    mcts_value: ndarr_f32
    reward: ndarr_f32


@frozen(kw_only=True)
class TrainingData:
    """
    Similar to TrajectoryState, but in a form more convenient for network training.
    A list of TrainingData is a complete training batch, it stores multiple continous
    sections of game trajectories stacked in the batch dimension.
    The list length is unroll_length, and the stacked height inside each tensor of a
    single TrainingData is batch_size.

                    timestep / list index
    list[TrainingData]    0 1 2 3 4
                         |         |
                    trajectory move number
    trajectory a  0 1 2 3|4 5 6 7 8|9 ...      ^
    trajectory b         |0 1 2 3 4|5 6 7 8 9  | batch_size
    trajectory c      0 1|2 3 4 5 6|           v
                          <------->
                        unroll_length

    Due to the nature of batching, each timestep is processed at once, but may contain
    different types of trajectory states (observation / own move, other player move).
    To select the correct behavior during training for each trajectory, this requires the
    use of boolean masks.
    """

    # masks:
    is_observation: Tensor

    observations: tuple[Tensor, ...]
    turn_status: Tensor
    action_onehot: Tensor
    target_policy: Tensor
    value_target: Tensor
    reward: Tensor

    @classmethod  # type: ignore [misc]
    @property
    @cast("functools._lru_cache_wrapper[tuple[Tensor, ...]]", functools.cache)
    def empty_observation(_) -> tuple[Tensor, ...]:
        return tuple(map(torch.zeros, C.game.instance.observation_shapes))

    @classmethod
    def from_trajectory_state(
        cls,
        ts: TrajectoryState,
        value_target: ndarr_f32,
        is_terminal: bool,
        cache: Optional[TensorCache] = None,
    ) -> TrainingData:
        if cache is None:
            cache = TensorCache()
        return cls(
            is_observation=cache.tensor(ts.observations is not None),
            observations=ts.observations
            if ts.observations is not None
            else cls.empty_observation,
            turn_status=cache.tensor(
                TurnStatus.TERMINAL_STATE.target_index if is_terminal else ts.turn_status,
                dtype=torch.long,
            ),
            action_onehot=cache.onehot(
                ts.action,
                C.game.instance.max_num_actions,
            ),
            target_policy=torch.tensor(ts.target_policy),
            value_target=torch.tensor(value_target, dtype=torch.float32),
            reward=torch.tensor(ts.reward),
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
            if name == "observations":
                stacked_data = tuple(map(torch.stack, zip(*data)))
            else:
                if not (name.startswith("is_") or name == "turn_status"):
                    data = torch.atleast_1d(data)  # type: ignore [no-untyped-call]
                    # only keep this data 0D:
                    # - is_*: boolean masks, processed specially in the training
                    # - turn_status: cross entropy loss takes class indices directly
                stacked_data = torch.stack(data)
            stacked_fields[name] = stacked_data
        return cls(**stacked_fields)  # type: ignore [arg-type]
