import math
from typing import TypeAlias
from collections.abc import Callable

from mcts import Node
from config import C

selection_fn: TypeAlias = Callable[[Node], float]


def ucb_score(node: Node) -> float:
    prior_scale = (
        (
            math.log(
                (
                    node.parent.visit_count
                    + C.mcts.fn.selection.ucb_score.prior_log_scale_base
                    + 1
                )
                / C.mcts.fn.selection.ucb_score.prior_log_scale_base
            )
            + C.mcts.fn.selection.ucb_score.prior_log_scale_init
        )
        * math.sqrt(node.parent.visit_count)
        / (node.visit_count + 1)
    )
    prior_score = node.prior * prior_scale
    if not node.is_expanded:
        return prior_score
    value_score = node.reward + node.value * C.train.discount_factor
    return value_score + prior_score


def from_prior(node: Node) -> float:
    return node.prior / (node.visit_count + 1)
