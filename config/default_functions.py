from collections.abc import Sequence

from mcts import Node
from config import config as C


def target_policy_visit_count(node: Node) -> Sequence[float]:
    visit_counts = [child.visit_count for child in node.children]
    return softmax(visit_counts, C.mcts.target_policy_visit_count.softmax_temp)
