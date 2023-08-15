from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, TypeVar, cast
from operator import itemgetter
from collections.abc import Callable, Iterable, Iterator

import numpy as np

from util import NaNWarning, ndarr_f32

if TYPE_CHECKING:
    from mcts import Node


def softmax(values: Iterable[float], temp: float = 1.0) -> ndarr_f32:
    vals = np.array(values, dtype=np.float32)
    exp = np.exp(vals / temp)
    result = exp / exp.sum()
    if np.isnan(result).any():
        warnings.warn(
            "NaN values encountered in softmax!\n"
            + f"{len(vals)} values, temperature: {temp}\n"
            + "input, temp applied, result\n"
            + np.array2string(
                np.column_stack((vals, vals / temp, result)),
                max_line_width=None,
                precision=4,
                suppress_small=True,
            ),
            NaNWarning,
            stacklevel=2,
        )
    return cast(ndarr_f32, result)


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


def get_visit_counts(node: Node) -> Iterator[int]:
    yield from map_actions_callback(node, lambda _, child: child.visit_count, lambda _: 0)


class SoftmaxTemp:
    def __init__(self, softmax_temp: float = 1):
        self.temp = softmax_temp
