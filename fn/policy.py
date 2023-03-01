from typing import TypeAlias
from collections.abc import Callable, Sequence

import numpy as np

from mcts import Node
from config import C

from .util import SoftmaxTemp, softmax

PolicyFn: TypeAlias = Callable[[Node], Sequence[float]]


def assert_fn_type(fn: PolicyFn) -> None:
    """For typechecking."""
    pass


def from_visit_count(node: Node) -> Sequence[float]:
    visit_counts = [child.visit_count for child in node.children]
    return softmax(visit_counts, C.mcts.fn.policy.from_visit_count.softmax_temp)


def from_visit_count_expanded(node: Node) -> Sequence[float]:
    visit_counts, idx = get_values_where_expanded(node.children, lambda n: n.visit_count)
    probs = np.full(len(node.children), 0.0)
    probs[idx] = softmax(visit_counts, C.mcts.fn.policy.from_visit_count.softmax_temp)
    return probs


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
