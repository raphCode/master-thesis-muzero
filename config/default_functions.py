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
    print(dist)
    print(temp)
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


def target_policy_visit_count(node: Node) -> Sequence[float]:
    visit_counts = [child.visit_count for child in node.children]
    return softmax(visit_counts, C.mcts.target_policy_visit_count.softmax_temp)


def selection_score_muzero_ucb(node: Node) -> float:
    prior_scale = (
        math.log(
            (
                node.parent.visit_count
                + C.mcts.selection_score_muzero_ucb.prior_log_scale_base
                + 1
            )
            / C.mcts.selection_score_muzero_ucb.prior_log_scale_base
        )
        + C.mcts.selection_score_muzero_ucb.prior_log_scale_init
    ) * math.sqrt(node.parent.visit_count / (node.visit_count + 1))
    prior_score = node.prior * prior_scale
    if not node.is_expanded:
        return prior_score
    value_score = node.reward + node.value * C.train.discount_factor
    return value_score + prior_score

def sane_selection_score(node: Node) -> float:
    prior_score = (node.prior +C.mcts.sane_selection_score.equalisation_prior)/ (node.visit_count+1)
    if not node.is_expanded:
        return prior_score
    diff_value = (node.value - node.parent.value ) * C.mcts.sane_selection_score.value_scale
    value_weight = math.sqrt(node.visit_count+1)
    value_score = node.reward + node.value * C.train.discount_factor
    return value_weight * value_score + prior_score
