from enum import IntEnum, auto


class PlayerType(IntEnum):
    """
    Reprents whose turn it is as seen by a specific player.
    This is important in the recorded trajectories and in the tree search.
    The chance player models randomness in a game by taking a random action.
    """
    Own = auto()
    Other = auto()
    Chance = auto()

