from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias
from collections.abc import Callable

import numpy as np
from attrs import frozen

from mcts import Node
from config.impl import assert_callable

if TYPE_CHECKING:
    from .policy import PolicyFn

rng = np.random.default_rng()

ActionFn: TypeAlias = Callable[[Node], int]
# TODO: legal actions mask


@frozen
class Counters:
    move: int
    game: int
    move_total: int


"""
def action(
    node: Node, legal_actions_mask: Sequence[bool], root_node: bool, counters: Counters
) -> Sequence[float]:
    pass
"""

# questions:
# - root node is what? always for fn/action, but what for fn/seletion in intermediate moves? we known intermediate moves because the current player id is not ours - but we need to know this one...
# legal actions part of Node? probably yes

# we have a flag whether the node is for action decision
# that may be a parameter to the selection fn call
# or better, call a custom init fn on the action decision node(s) so it can set up a flag, exploration noise etc.


def assert_fn_type(fn: ActionFn) -> None:
    """For typechecking."""
    pass


class DrawFromPolicyFn:
    """
    Repurposes a policy fn by drawing an action from the returned distribution.
    """

    def __init__(self, policy_fn: PolicyFn):
        assert_callable(policy_fn)
        self.policy_fn = policy_fn

    def __call__(self, node: Node) -> int:
        probs = self.policy_fn(node)
        return rng.choice(len(probs), p=probs)
