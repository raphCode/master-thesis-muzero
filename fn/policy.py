from typing import TypeAlias
from collections.abc import Callable, Sequence

import numpy as np

from mcts import Node
from config import C

from .util import softmax, get_values_where_expanded

policy_fn: TypeAlias = Callable[[Node], Sequence[float]]


def assert_fn_type(fn: policy_fn) -> None:
    pass


def from_visit_count(node: Node) -> Sequence[float]:
    visit_counts = [child.visit_count for child in node.children]
    return softmax(visit_counts, C.mcts.fn.policy.from_visit_count.softmax_temp)


def from_visit_count_expanded(node: Node) -> Sequence[float]:
    visit_counts, idx = get_values_where_expanded(node.children, lambda n: n.visit_count)
    probs = np.full(len(node.children), 0.0)
    probs[idx] = softmax(visit_counts, C.mcts.fn.policy.from_visit_count.softmax_temp)
    return probs


def from_value_expanded(node: Node) -> Sequence[float]:
    values, idx = get_values_where_expanded(
        node.children, lambda n: n.reward + n.value * C.train.discount_factor
    )
    policy = np.full(len(node.children), 0.0)
    policy[idx] = softmax(values, C.mcts.fn.policy.from_value.softmax_temp, norm=False)
    return policy
