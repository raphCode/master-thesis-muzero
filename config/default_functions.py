import math

import numpy as np

from mcts import Node
from config import config as C

from collections.abc import Collection
rng = np.random.default_rng()


def softmax(dist: Collection[float], temp: float = 1.0, norm:bool=True) -> np.ndarray:
    dist=np.array(dist)
    if norm:
        temp *= dist.sum()
    exp= np.exp(dist/temp)
    return exp / exp.sum()


def default_reward(rewards: tuple[float], player_id: int) -> float:
    return rewards[player_id]


def no_teammate(pid_a: int, pid_b: int) -> bool:
    return False


# TODO: rename
def greedy_node_action(node: Node, move_number: int) -> int:
    visit_counts = [child.visit_count for child in node.children]
    temp = C.mcts.softmax_temp if move_number < 30 else 0
    return rng.choice(C.game.instance.max_num_actions, p=softmax(visit_counts, temp))


def muzero_node_target_policy(node: Node) -> list[float]:
    visit_counts = [child.visit_count for child in node.children]
    return softmax(visit_counts, C.mcts.softmax_temp)


def muzero_node_ucb_selection_score(node: Node, debug:bool=False) -> float:
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
    if debug:
        return value_score, prior_score
    return value_score + prior_score
