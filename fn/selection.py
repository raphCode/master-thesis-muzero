import math
from typing import TypeAlias
from collections.abc import Callable

from mcts import Node
from config import C

SelectionFn: TypeAlias = Callable[[Node], int]


def assert_fn_type(fn: SelectionFn) -> None:
    """For typechecking."""
    pass


def map_actions_best(
    node: Node,
    score_fn_child: Callable[[float, Node], float],  # prior, child -> score
    score_fn_nochild: Callable[[float], float],  # prior -> score
) -> int:
    """
    Call the score functions for all actions and return the action with the highest score.
    One score function is called for each expanded action (with child Nodes), the other is
    called for unexpanded ones.
    """
    best = tuple()  # type: tuple[float, int]  # type: ignore [assignment]
    for action, prior in enumerate(node.probs):
        if action in node.children:
            best = max(best, (score_fn_child(prior, node.children[action]), action))
        else:
            best = max(best, (score_fn_nochild(prior), action))
    return best[1]


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

        return map_actions_best(node, child_score, lambda prior: prior * prior_scale_half)


assert_fn_type(UCBScore())


def from_prior(node: Node) -> int:
    return map_actions_best(
        node, lambda prior, child: prior / (child.visit_count + 1), lambda prior: prior
    )


assert_fn_type(from_prior)
