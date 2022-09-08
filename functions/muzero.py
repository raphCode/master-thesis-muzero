import math
from collections.abc import Callable, Iterable, Sequence

import numpy as np

from mcts import Node
from config import config as C

from .util import softmax

rng = np.random.default_rng()


def visit_count_action(node: Node, move_number: int) -> int:
    visit_counts = [child.visit_count for child in node.children]
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
    return rng.choice(C.game.instance.max_num_actions, p=softmax(visit_counts, temp))


def visit_count_target(node: Node) -> Sequence[float]:
    visit_counts = [child.visit_count for child in node.children]
    return softmax(visit_counts, C.mcts.visit_count_target.softmax_temp)


def ucb_score(node: Node) -> float:
    prior_scale = (
        (
            math.log(
                (node.parent.visit_count + C.mcts.muzero_ucb.prior_log_scale_base + 1)
                / C.mcts.muzero_ucb.prior_log_scale_base
            )
            + C.mcts.muzero_ucb.prior_log_scale_init
        )
        * math.sqrt(node.parent.visit_count)
        / (node.visit_count + 1)
    )
    prior_score = node.prior * prior_scale
    if not node.is_expanded:
        return prior_score
    value_score = node.reward + node.value * C.train.discount_factor
    return value_score + prior_score
