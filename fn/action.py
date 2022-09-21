import numpy as np

from mcts import Node
from config import config as C

from .util import softmax, get_values_where_expanded

rng = np.random.default_rng()


def from_visit_count(node: Node, move_number: int) -> int:
    visit_counts = [child.visit_count for child in node.children]
    temp = np.interp(
        move_number,
        (
            C.mcts.fn.action.from_visit_count.num_moves_start,
            C.mcts.fn.action.from_visit_count.num_moves_end,
        ),
        (
            C.mcts.fn.action.from_visit_count.softmax_temp_start,
            C.mcts.fn.action.from_visit_count.softmax_temp_end,
        ),
    )
    return rng.choice(C.game.instance.max_num_actions, p=softmax(visit_counts, temp))


def from_visit_count_expanded(node: Node, move_number: int) -> int:
    visit_counts, idx = get_values_where_expanded(node.children, lambda n: n.visit_count)
    temp = np.interp(
        move_number,
        (
            C.mcts.fn.action.from_visit_count.num_moves_start,
            C.mcts.fn.action.from_visit_count.num_moves_end,
        ),
        (
            C.mcts.fn.action.from_visit_count.softmax_temp_start,
            C.mcts.fn.action.from_visit_count.softmax_temp_end,
        ),
    )
    probs = np.full(len(node.children), 0.0)
    probs[idx] = softmax(visit_counts, temp)
    return rng.choice(C.game.instance.max_num_actions, p=probs)
