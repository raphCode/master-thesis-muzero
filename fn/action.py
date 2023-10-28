from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias
from collections.abc import Callable

import numpy as np

from mcts import Node
from config.impl import assert_callable

if TYPE_CHECKING:
    from .policy import PolicyFn

rng = np.random.default_rng()

ActionFn: TypeAlias = Callable[[Node], int]


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


def highest_visit_count(node: Node) -> int:
    return max(node.children.keys(), key=lambda a: node.children[a].visit_count)


assert_fn_type(highest_visit_count)
