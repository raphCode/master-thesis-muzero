from enum import IntEnum, auto
from collections import namedtuple


class PlayerType(IntEnum):
    """
    Reprents whose turn it is as seen by a specific player.
    This is important in the recorded trajectories and in the tree search.
    The chance player models randomness in a game by taking a random action.
    """

    Own = auto()
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
