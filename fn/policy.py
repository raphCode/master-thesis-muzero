from typing import TypeAlias
from collections.abc import Callable, Sequence

import numpy as np

from mcts import Node

from .util import SoftmaxTemp, softmax, get_visit_counts

PolicyFn: TypeAlias = Callable[[Node], Sequence[float]]


def assert_fn_type(fn: PolicyFn) -> None:
    """For typechecking."""
    pass


def from_visit_counts(node: Node) -> Sequence[float]:
    """
    Linearly scales the child visit counts into a probability distribution.
    """
    assert len(node.children) > 0
    visit_counts = np.fromiter(get_visit_counts(node), dtype=int)
    return visit_counts / visit_counts.sum()  # type: ignore [no-any-return]


assert_fn_type(from_visit_counts)


class FromExpandedValues(SoftmaxTemp):
    """
    Creates a policy by applying a softmax to the child mcts value estimates.
    Unexpanded actions are excluded from the softmax und become a policy value of zero.
    """

    def __call__(self, node: Node) -> Sequence[float]:
        assert len(node.children) > 0
        values = [child.value for child in node.children.values()]
        policy = np.full(len(node.children), 0.0)
        policy[node.children.keys()] = softmax(values, self.temp)
        return policy  # type: ignore [return-value]


assert_fn_type(FromExpandedValues())
