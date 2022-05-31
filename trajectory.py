from enum import IntEnum, auto
from typing import List, Tuple, NamedTuple

import numpy as np


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


class TrajectoryState(NamedTuple):
    observation: torch.Tensor
    latent_rep: torch.Tensor
    beliefs: torch.Tensor
    player_type: PlayerType
    action: int
    target_policy: torch.Tensor
    value: float  # either mcts estimate (trajectory) or value target (training data)
    reward: float


rng = np.random.default_rng()
discounts = np.concatenate(
    ([1], np.cumprod(np.full(C.param.n_step_return - 1, C.param.reward_discount)))
)


class ReplayBuffer:
    lens = deque[int]
    data = deque[List[TrajectoryState]]

    def __init__(self, size: int):
        self.lens = deque(maxlen=size)
        self.data = deque(maxlen=size)

    def add_trajectory(self, traj: List[TrajectoryState], game_terminated: bool):
        *_, rewards = zip(*traj)
        rewards = np.array(rewards)

        train_data = []
        for n, ts in enumerate(traj):
            nstep_idx = min(len(traj), n + C.param.n_step_return)
            if nstep_idx == len(traj) and game_terminated:
                v = 0
            else:
                nstep_idx -= 1
                v = traj[nstep_idx].value * discounts[nstep_idx - n]

            v += np.inner(rewards[n:nstep_idx], discounts[: nstep_idx - n])
            *data, _, reward = ts
            train_data.append(TrajectoryState(*data, v, reward))

        self.lens.append(len(train_data))
        self.data.append(train_data)

    def sample(self) -> List[List[TrajectoryState]]:
        lens = np.array(self.lens)
        probs = lens / lens.sum()
        batch = []
        batchsize = 0
        while batchsize < C.param.min_batchsize:
            self_idx, traj = rng.choice(self.data, p=probs)
            i = rng.integers(len(self_idx))
            end = self_idx.get(i + C.param.batch_continuous_rounds, len(traj))
            segment = traj[self_idx[i] : end + 1]
            batch.append(segment)
            batchsize += len(segment)
        return batch
