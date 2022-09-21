from collections.abc import Sequence

import numpy as np

from mcts import Node
from config import config as C

from .util import softmax, get_values_where_expanded


def from_visit_count(node: Node) -> Sequence[float]:
    visit_counts = [child.visit_count for child in node.children]
    return softmax(visit_counts, C.mcts.fn.policy.from_visit_count.softmax_temp)


def from_visit_count_expanded(node: Node) -> Sequence[float]:
    visit_counts, idx = get_values_where_expanded(node.children, lambda n: n.visit_count)
    probs = np.full(len(node.children), 0.0)
    probs[idx] = softmax(visit_counts, C.mcts.fn.policy.from_visit_count.softmax_temp)
    return probs
