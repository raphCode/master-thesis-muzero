from typing import TypeAlias, cast
from collections.abc import Callable

import numpy as np

from mcts import Node, StateNode, TurnStatus
from util import ndarr_f32
from config import C

from .util import SoftmaxTemp, softmax, get_visit_counts

PolicyFn: TypeAlias = Callable[[Node], ndarr_f32]


def assert_fn_type(fn: PolicyFn) -> None:
    """For typechecking."""
    pass


def from_visit_counts(node: Node) -> ndarr_f32:
    """
    Linearly scales the child visit counts into a probability distribution.
    """
    assert len(node.children) > 0
    visit_counts = np.fromiter(get_visit_counts(node), dtype=int)
    return cast(ndarr_f32, (visit_counts / visit_counts.sum()).astype(np.float32))


assert_fn_type(from_visit_counts)


class FromExpandedValues(SoftmaxTemp):
    """
    Creates a policy by applying a softmax to the child mcts value estimates.
    Unexpanded actions are excluded from the softmax und become a policy value of zero.
    """

    def __call__(self, node: Node) -> ndarr_f32:
        assert isinstance(node, StateNode)
        assert node.player is not TurnStatus.CHANCE_PLAYER
        assert len(node.children) > 0
        values = [
            child.normalized_reward[node.player] + child.normalized_value[node.player]
            for child in node.children.values()
        ]
        policy = np.zeros(C.game.instance.max_num_actions, dtype=np.float32)
        policy[list(node.children.keys())] = softmax(values, self.temp)
        return policy


assert_fn_type(FromExpandedValues())
