from typing import Any, TypeVar
from operator import itemgetter
from collections.abc import Callable, Iterable, Iterator, Sequence

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


T = TypeVar("T")


def argmax(values: Iterable[Any]) -> int:
    return max(enumerate(values), key=itemgetter(1))[0]


def map_actions_callback(
    node: Node,
    callback_child: Callable[[float, Node], T],  # prior, child
    callback_nochild: Callable[[float], T],  # prior
) -> Iterator[T]:
    """
    Map each possible action to one of the callbacks and yield the results.
    The callback is chosen based on wheter the action is expanded, i.e. has a child node.
    """
    for action, prior in enumerate(node.probs):
        if action in node.children:
            yield callback_child(prior, node.children[action])
        else:
            yield callback_nochild(prior)
