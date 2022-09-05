import functools
from enum import IntEnum
from typing import Deque, Optional
from collections import deque
from collections.abc import Sequence

import attrs
import numpy as np
import torch
import torch.nn.functional as F
from attrs import frozen

from config import config as C


class PlayerType(IntEnum):
    """
    Reprents whose turn it is as seen by a specific player.
    This is important in the recorded trajectories and in the tree search.
    The chance player models randomness in a game by taking a random action.
    """

    Self = 0
    Chance = 1
    Opponent = 2
    Teammate = 3


@frozen(kw_only=True)
class ObservationInfo:
    observation: tuple[torch.Tensor]
    prev_beliefs: torch.Tensor  # from previous state, for representation inference


@frozen(kw_only=True)
class LatentInfo:
    latent_rep: Optional[torch.Tensor]
    beliefs: Optional[torch.Tensor]


@frozen(kw_only=True)
class TrajectoryState:
    info: ObservationInfo | LatentInfo
    player_type: PlayerType
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


rng = np.random.default_rng()


class ReplayBuffer:
    lens = Deque[int]
    data = Deque[list[TrajectoryState]]

    def __init__(self):
        self.lens = deque(maxlen=C.train.replay_buffer_size)
        self.data = deque(maxlen=C.train.replay_buffer_size)
        self.discounts = np.concatenate(
            ([1], np.cumprod(np.full(C.train.n_step_return - 1, C.train.discount_factor)))
        )
        self.empty_observation = tuple(
            torch.zeros(s, dtype=torch.float) for s in C.game.instance.observation_shapes
        )
        self.empty_latent_rep = torch.zeros(C.nets.latent_rep_shape)

    def add_trajectory(self, traj: list[TrajectoryState], game_terminated: bool):
        int64t = functools.partial(torch.tensor, dtype=torch.int64)
        floatt = functools.partial(torch.tensor, dtype=torch.float)

        rewards = np.array([ts.reward for ts in traj])

        train_data = []
        for n, ts in enumerate(traj):
            nstep_idx = min(len(traj), n + C.train.n_step_return)
            if nstep_idx == len(traj) and game_terminated:
                value_target = 0
            else:
                nstep_idx -= 1
                value_target = traj[nstep_idx].value * self.discounts[nstep_idx - n]

            value_target += np.inner(
                rewards[n:nstep_idx], self.discounts[: nstep_idx - n]
            )
            is_obs = isinstance(ts.info, ObservationInfo)
            train_data.append(
                TrainingData(
                    is_observation=torch.tensor(is_obs),
                    is_data=torch.tensor(True),
                    observation=ts.info.observation if is_obs else self.empty_observation,
                    latent_rep=self.empty_latent_rep if is_obs else ts.info.latent_rep,
                    beliefs=ts.info.prev_beliefs if is_obs else ts.info.beliefs,
                    player_type=int64t(ts.player_type),
                    action_onehot=F.one_hot(
                        int64t(ts.action), C.game.instance.max_num_actions
                    ),
                    target_policy=floatt(ts.target_policy),
                    value_target=floatt(value_target),
                    reward=floatt(ts.reward),
                )
            )

        self.lens.append(len(train_data))
        self.data.append(train_data)

    def sample(self) -> list[list[TrajectoryState]]:
        lens = np.array(self.lens)
        probs = lens / lens.sum()
        batch = []
        data = np.empty(len(self.data), dtype=object)
        data[:] = self.data
        for traj in rng.choice(data, size=C.train.batch_num_games, p=probs):
            i = rng.integers(len(traj))
            batch.append(traj[i : i + C.train.batch_game_size])
        return batch

    def __len__(self):
        return len(self.data)
