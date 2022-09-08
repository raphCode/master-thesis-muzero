from collections.abc import Callable, Iterable, Sequence

import numpy as np

from mcts import Node
from config import config as C


def standard_reward(rewards: tuple[float], player_id: int) -> float:
    return rewards[player_id]


def no_teammate(pid_a: int, pid_b: int) -> bool:
    return False
