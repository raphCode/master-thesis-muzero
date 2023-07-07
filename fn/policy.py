from typing import TypeAlias, cast
from collections.abc import Callable

import numpy as np

from mcts import Node
from util import ndarr_f64
from config import C

from .util import SoftmaxTemp, softmax, get_visit_counts

PolicyFn: TypeAlias = Callable[[Node], ndarr_f64]


def assert_fn_type(fn: PolicyFn) -> None:
    """For typechecking."""
    pass


def from_visit_counts(node: Node) -> ndarr_f64:
    """
    Linearly scales the child visit counts into a probability distribution.
    """
    assert len(node.children) > 0
    visit_counts = np.fromiter(get_visit_counts(node), dtype=int)
    return cast(ndarr_f64, visit_counts / visit_counts.sum())


assert_fn_type(from_visit_counts)


class FromExpandedValues(SoftmaxTemp):
    """
    Creates a policy by applying a softmax to the child mcts value estimates.
    Unexpanded actions are excluded from the softmax und become a policy value of zero.
    """

    def __call__(self, node: Node) -> ndarr_f64:
        assert len(node.children) > 0
        values = [
            child.normalized_reward + child.normalized_value
            for child in node.children.values()
        ]
        policy = np.full(C.game.instance.max_num_actions, 0.0)
        policy[list(node.children.keys())] = softmax(values, self.temp)
        return policy


assert_fn_type(FromExpandedValues())
