from typing import TypeAlias
from collections.abc import Callable, Sequence

import numpy as np

from mcts import Node
from config import C

from .util import softmax, get_values_where_expanded

rng = np.random.default_rng()

ActionFn: TypeAlias = Callable[[Node], int]


def assert_fn_type(fn: ActionFn) -> None:
    """For typechecking."""
    pass


def _interp_softmax_temp(move_number: int, cfg_name: str) -> float:
    cfg = getattr(C.mcts.fn.action, cfg_name)
    return np.interp(
        move_number,
        (
            cfg.move_num_start,
            cfg.move_num_end,
        ),
        (
            cfg.softmax_temp_start,
            cfg.softmax_temp_end,
        ),
    )


def from_visit_count(node: Node, move_number: int) -> int:
    visit_counts = [child.visit_count for child in node.children]
    temp = _interp_softmax_temp(move_number, "from_visit_count")
    return rng.choice(C.game.instance.max_num_actions, p=softmax(visit_counts, temp))


def from_visit_count_expanded(node: Node, move_number: int) -> int:
    visit_counts, idx = get_values_where_expanded(node.children, lambda n: n.visit_count)
    temp = _interp_softmax_temp(move_number, "from_visit_count")
    probs = np.full(len(node.children), 0.0)
    probs[idx] = softmax(visit_counts, temp)
    return rng.choice(C.game.instance.max_num_actions, p=probs)
