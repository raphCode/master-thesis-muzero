import math

from mcts import Node
from config import config as C


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
