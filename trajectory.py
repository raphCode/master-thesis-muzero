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
    old_beliefs: torch.Tensor  # old beliefs prior to the representation inference
    dyn_beliefs: torch.Tensor  # beliefs after the dynamics inference
    player_type: PlayerType
    action: int
    target_policy: torch.Tensor
    value: float  # either mcts estimate (trajectory) or value target (training data)
    reward: float


rng = np.random.default_rng()
discounts = np.concatenate(
    ([1], np.cumprod(np.full(C.param.n_step_return - 1, C.param.discount_factor)))
)


class ReplayBuffer:
    lens = deque[int]
    data = deque[List[TrajectoryState]]

    def __init__(self, size: int):
        self.lens = deque(maxlen=size)
        self.data = deque(maxlen=size)

    def add_trajectory(self, traj: List[TrajectoryState], game_terminated: bool):
        rewards = np.fromiter((ts.reward for ts in traj), dtype=float, count=len(traj))

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
        for _ in range(C.param.batch_num_games):
            traj = rng.choice(self.data, p=probs)
            i = rng.integers(len(traj))
            batch.append(traj[i : i + C.param.batch_game_size])
        return batch
