from enum import IntEnum, auto
from typing import List
from collections import namedtuple


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


TrajectoryState = namedtuple(
    "TrajectoryState",
    [
        "observation",
        "beliefs",
        "player_onehot",
        "action",
        "rewards",
        "target_policy",
    ],
)


rng = np.random.default_rng()


class ReplayBuffer:
    lens = deque[int]
    trajs = deque[List[TrajectoryState]]

    def __init__(self, size: int):
        self.lens = deque(maxlen=size)
        self.trajs = deque(maxlen=size)

    def add_trajectory(self, traj: List[TrajectoryState]):
        self.trajs.append(traj)
        self.lens.append(len(traj))

    def sample(self, size: int) -> List[TrajectoryState]:
        lens = np.array(self.lens)
        traj = rng.choice(self.trajs, p=lens / lens.sum())
        idx = rng.integers(len(traj))
        # TODO: Batch from different games, always start batch at own moves
        return traj[idx : idx + C.param.batchsize]
