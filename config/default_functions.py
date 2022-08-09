import math
from collections.abc import Sequence

import numpy as np

from mcts import Node
from config import config as C

rng = np.random.default_rng()


def softmax(dist: Sequence[float], temp: float = 1.0, norm: bool = True) -> np.ndarray:
    dist = np.array(dist)
    if norm:
        temp *= dist.sum()
    exp = np.exp(dist / temp)
    return exp / exp.sum()


def default_reward(rewards: tuple[float], player_id: int) -> float:
    return rewards[player_id]


def no_teammate(pid_a: int, pid_b: int) -> bool:
    return False


def action_visit_count(node: Node, move_number: int) -> int:
    visit_counts = [child.visit_count for child in node.children]
    temp = np.interp(
        move_number,
        (
            C.mcts.action_visit_count.num_moves_start,
            C.mcts.action_visit_count.num_moves_end,
        ),
        (
            C.mcts.action_visit_count.softmax_temp_start,
            C.mcts.action_visit_count.softmax_temp_end,
        ),
    )
    return rng.choice(C.game.instance.max_num_actions, p=softmax(visit_counts, temp))


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
    ) * math.sqrt(node.parent.visit_count / (node.visit_count + 1))
    prior_score = node.prior * prior_scale
    if not node.is_expanded:
        return prior_score
    value_score = node.reward + node.value * C.train.discount_factor
    return value_score + prior_score
