from collections.abc import Callable, Iterable, Sequence

import numpy as np

from mcts import Node
from config import config as C

from .util import softmax, get_values_where_expanded

rng = np.random.default_rng()


def visit_count_action(node: Node, move_number: int) -> int:
    visit_counts, idx = get_values_where_expanded(node.children, lambda n: n.visit_count)
    temp = np.interp(
        move_number,
        (
            C.mcts.visit_count_action.num_moves_start,
            C.mcts.visit_count_action.num_moves_end,
        ),
        (
            C.mcts.visit_count_action.softmax_temp_start,
            C.mcts.visit_count_action.softmax_temp_end,
        ),
    )
    probs = np.full(len(node.children), 0.0)
    probs[idx] = softmax(visit_counts, temp)
    return rng.choice(C.game.instance.max_num_actions, p=probs)


def visit_count_target(node: Node) -> Sequence[float]:
    visit_counts, idx = get_values_where_expanded(node.children, lambda n: n.visit_count)
    probs = np.full(len(node.children), 0.0)
    probs[idx] = softmax(visit_counts, C.mcts.visit_count_target.softmax_temp)
    return probs
