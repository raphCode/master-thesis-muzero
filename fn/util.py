from collections.abc import Callable, Iterable, Sequence

import numpy as np

from mcts import Node


def softmax(dist: Sequence[float], temp: float = 1.0, norm: bool = True) -> np.ndarray:
    dist = np.array(dist)
    if norm:
        temp *= dist.sum()
    exp = np.exp(dist / temp)
    return exp / exp.sum()


def get_values_where_expanded(
    nodes: Iterable[Node], value_fn: Callable[[Node], float | int]
) -> tuple[np.ndarray, np.ndarray]:
    indices = []
    values = []
    for n, node in enumerate(nodes):
        if node.is_expanded:
            values.append(value_fn(node))
            indices.append(n)
    return np.array(values), np.array(indices)
