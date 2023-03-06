from typing import TypeAlias
from collections.abc import Callable

import numpy as np

from mcts import Node
from config.impl import assert_callable

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
