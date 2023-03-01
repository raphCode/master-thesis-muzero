from typing import Any
from collections.abc import Callable, Iterable, Sequence

import numpy as np

from mcts import Node


def softmax(
    values: Sequence[float], temp: float = 1.0, norm: bool = True
) -> np.ndarray[Any, np.dtype[np.float64]]:
    dist = np.array(values)
    if norm and (dist_sum := dist.sum()) > 0:
        temp *= dist_sum
    exp = np.exp(dist / temp)
    return exp / exp.sum()  # type: ignore [no-any-return]


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
