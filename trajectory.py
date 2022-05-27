from enum import IntEnum, auto
from typing import List, Tuple, NamedTuple


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
    beliefs: torch.Tensor
    player_type: PlayerType
    action: int
    target_policy: torch.Tensor
    mcts_value: float
    # The following value actually refers to the next state after the action transition
    reward: float


rng = np.random.default_rng()


class ReplayBuffer:
    lens = deque[int]
    data = deque[Tuple[List[int], List[TrajectoryState]]]

    def __init__(self, size: int):
        self.lens = deque(maxlen=size)
        self.data = deque(maxlen=size)

    def add_trajectory(self, traj: List[TrajectoryState]):
        self_idx = [n for n, ts in enumerate(traj) if ts.player_type == PlayerType.Self]
        self.lens.append(len(self_idx))
        self.data.append((self_idx, traj))

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
