import math
from typing import TYPE_CHECKING, TypeAlias
from collections.abc import Callable

import numpy as np

from mcts import Node, StateNode, TurnStatus, TerminalNode
from config import C

from .util import argmax, map_actions_callback

SelectionFn: TypeAlias = Callable[[Node], int]

rng = np.random.default_rng()


def assert_fn_type(fn: SelectionFn) -> None:
    """For typechecking."""
    pass


class pUCTscore:
    def __init__(self, c1: float = 1.25, c2: float = 1000):
        self.c1 = c1
        self.c2 = c2

    def __call__(self, node: Node) -> int:
        assert isinstance(node, StateNode)
        assert node.player is not TurnStatus.CHANCE_PLAYER
        prior_scale_half = (
            self.c1 + math.log((node.visit_count + self.c2 + 1) / self.c2)
        ) * math.sqrt(node.visit_count)

        def child_score(prior: float, child: Node) -> float:
            if TYPE_CHECKING:
                assert isinstance(node, StateNode)
                assert node.player is not TurnStatus.CHANCE_PLAYER
            prior_score = prior * prior_scale_half / (child.visit_count + 1)
            value_score = (
                child.normalized_reward[node.player]
                + child.normalized_value[node.player] * C.training.discount_factor
            )
            return float(value_score.clip(0, 1)) + prior_score

        uct_scores = np.fromiter(
            map_actions_callback(
                node, child_score, lambda prior: prior * prior_scale_half
            ),
            dtype=np.float64,
        )
        if node.mask is not None:
            uct_scores[~node.mask] = float("-inf")
        return argmax(uct_scores)


assert_fn_type(pUCTscore())


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


class SwitchOnNodeType:
    """
    Switches between selection fns depending on the node (chance / terminal / player)
    Effectively makes other selection fns compatible with chance and terminal nodes.
    """

    def __init__(
        self,
        normal_selection_fn: SelectionFn,
        chance_selection_fn: SelectionFn = from_prior_deterministic,
        terminal_selection_fn: SelectionFn = from_prior_deterministic,
    ):
        self.normal_fn = normal_selection_fn
        self.chance_fn = chance_selection_fn
        self.terminal_fn = terminal_selection_fn

    def __call__(self, node: Node) -> int:
        if isinstance(node, TerminalNode):
            return self.terminal_fn(node)
        if isinstance(node, StateNode) and node.player is TurnStatus.CHANCE_PLAYER:
            return self.chance_fn(node)
        return self.normal_fn(node)


assert_fn_type(SwitchOnNodeType(from_prior_deterministic))
