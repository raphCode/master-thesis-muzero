import math
from typing import TypeAlias
from collections.abc import Callable

from mcts import Node
from config import C

selection_fn: TypeAlias = Callable[[Node], int]


def assert_fn_type(fn: selection_fn) -> None:
    pass


class UCBScore:
    def __init__(
        self, prior_log_scale_base: float = 5, prior_log_scale_init: float = 1.25
    ):
        self.base = prior_log_scale_base
        self.init = prior_log_scale_init

    def __call__(self, node: Node) -> list[float]:
        prior_scale_half = (
            math.log((node.visit_count + self.base + 1) / self.base) + self.init
        ) * math.sqrt(node.visit_count)
        result = []
        for action, prior in enumerate(node.probs):
            if action in node.children:
                child = node.children[action]
                prior_score = prior * prior_scale_half / (child.visit_count + 1)
                value_score = child.reward + child.value * C.training.discount_factor
                result.append(value_score + prior_score)
            else:
                result.append(prior * prior_scale_half)
        return result


assert_fn_type(UCBScore())


def from_prior(node: Node) -> list[float]:
    result = []
    for action, prior in enumerate(node.probs):
        vc = node.children[action].visit_count + 1 if action in node.children else 1
        result.append(prior / vc)
    return result


assert_fn_type(from_prior)
