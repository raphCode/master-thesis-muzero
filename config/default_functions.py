from collections.abc import Sequence

import numpy as np

from mcts import Node
from config import config as C

rng = np.random.default_rng()


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
