from collections.abc import Callable, Iterable, Sequence

import numpy as np

from mcts import Node
from config import config as C


def softmax(dist: Sequence[float], temp: float = 1.0, norm: bool = True) -> np.ndarray:
    dist = np.array(dist)
    if norm:
        temp *= dist.sum()
    exp = np.exp(dist / temp)
    return exp / exp.sum()
