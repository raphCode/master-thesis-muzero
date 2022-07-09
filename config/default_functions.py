import math

import numpy as np

from mcts import Node
from config import config as C


def default_reward(rewards: tuple[float], player_id: int) -> float:
    return rewards[player_id]


def no_teammate(pid_a: int, pid_b: int) -> bool:
    return False


def greedy_node_action(node: Node) -> int:
    _, node = max((child.visit_count, child) for child in node.children)
    return node


def muzero_node_target_policy(node: Node) -> list[float]:
    # softmax of visit counts
    visit_counts = [child.visit_count for child in node.children]
    exp = np.exp(visit_counts)
    return exp / exp.sum()


def muzero_node_ucb_selection_score(node: Node) -> float:
    prior_scale = (
        math.log(
            (node.parent.visit_count + C.mcts.ucb_prior_log_scale_base + 1)
            / C.mcts.ucb_prior_log_scale_base
        )
        + C.mcts.ucb_prior_log_scale_init
    ) * math.sqrt(node.parent.visit_count / node.visit_count)
    prior_score = node.prior * prior_scale
    value_score = node.reward + node.value * C.train.discount_factor
    return value_score + prior_score