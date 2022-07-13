from enum import IntEnum, auto
from typing import Deque
from collections import deque

import numpy as np
import torch
from attrs import evolve, frozen

from config import config as C


class PlayerType(IntEnum):
    """
    Reprents whose turn it is as seen by a specific player.
    This is important in the recorded trajectories and in the tree search.
    The chance player models randomness in a game by taking a random action.
    """

    Self = auto()
    Chance = auto()
    Opponent = auto()
    Teammate = auto()


@frozen(kw_only=True)
class TrajectoryState:
    observation: torch.Tensor
    latent_rep: torch.Tensor
    old_beliefs: torch.Tensor  # old beliefs prior to the representation inference
    dyn_beliefs: torch.Tensor  # beliefs after the dynamics inference
    player_type: PlayerType
    action: int
    target_policy: torch.Tensor
    value: float  # either mcts estimate (trajectory) or value target (training data)
    reward: float


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

    def add_trajectory(self, traj: list[TrajectoryState], game_terminated: bool):
        rewards = np.fromiter((ts.reward for ts in traj), dtype=float, count=len(traj))

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
            train_data.append(evolve(ts, value=value_target))

        self.lens.append(len(train_data))
        self.data.append(train_data)

    def sample(self) -> list[list[TrajectoryState]]:
        lens = np.array(self.lens)
        probs = lens / lens.sum()
        batch = []
        for _ in range(C.train.batch_num_games):
            traj = rng.choice(self.data, p=probs)
            i = rng.integers(len(traj))
            batch.append(traj[i : i + C.train.batch_game_size])
        return batch
