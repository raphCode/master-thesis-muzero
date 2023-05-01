import math
from typing import TypeAlias
from collections.abc import Callable

import numpy as np

from mcts import Node, StateNode
from config import C

from .util import argmax, map_actions_callback

SelectionFn: TypeAlias = Callable[[Node], int]

rng = np.random.default_rng()


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


def sample_from_prior(node: Node) -> int:
    """
    Draw a random action from the prior probability distribution.
    """
    return rng.choice(len(node.probs), p=node.probs)


assert_fn_type(sample_from_prior)


class SwitchOnChanceNodes:
    """
    Switches between two selection fns depending on the StateNode being a chance event.
    Effectively makes other selection fns compatible with chance nodes.
    Node types other than StateNode are handled with the normal function.
    """

    def __init__(
        self,
        normal_selection_fn: SelectionFn,
        chance_selection_fn: SelectionFn = from_prior_deterministic,
    ):
        self.normal_fn = normal_selection_fn
        self.chance_fn = chance_selection_fn

    def __call__(self, node: Node) -> int:
        if (
            isinstance(node, StateNode)
            and node.current_player == C.game.instance.chance_player_id
        ):
            return self.chance_fn(node)
        return self.normal_fn(node)


assert_fn_type(SwitchOnChanceNodes(from_prior_deterministic))
