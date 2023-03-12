import math
from typing import TypeAlias
from collections.abc import Callable

from mcts import Node
from config import C

from .util import argmax, map_actions_callback

SelectionFn: TypeAlias = Callable[[Node], int]


def assert_fn_type(fn: SelectionFn) -> None:
    """For typechecking."""
    pass


class UCBScore:
    def __init__(
        self, prior_log_scale_base: float = 5, prior_log_scale_init: float = 1.25
    ):
        self.base = prior_log_scale_base
        self.init = prior_log_scale_init

    def __call__(self, node: Node) -> int:
        prior_scale_half = (
            math.log((node.visit_count + self.base + 1) / self.base) + self.init
        ) * math.sqrt(node.visit_count)

        def child_score(prior: float, child: Node) -> float:
            prior_score = prior * prior_scale_half / (child.visit_count + 1)
            value_score = child.reward + child.value * C.training.discount_factor
            return value_score + prior_score

        return argmax(
            map_actions_callback(
                node, child_score, lambda prior: prior * prior_scale_half
            )
        )


assert_fn_type(UCBScore())


def from_prior_deterministic(node: Node) -> int:
    """
    Deterministic selection according to prior probability distribution.
    No randomness is involved, which reduces variance.
    """
    return argmax(
        map_actions_callback(
            node,
            lambda prior, child: prior / (child.visit_count + 1),
            lambda prior: prior,
        )
    )


assert_fn_type(from_prior_deterministic)
