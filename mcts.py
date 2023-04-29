from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, cast
from collections.abc import Iterable, MutableMapping

import torch
import torch.nn.functional as F
from torch import Tensor

from config import C
from networks.bases import Networks

if TYPE_CHECKING:
    # only needed for type annotations, can't import uncondionally due to import cycles
    from fn.selection import SelectionFn


class Node(ABC):
    """
    A Node represents a game state in the monte carlo search tree.

    Contrary to the MuZero original implementation, nodes are always expanded and children
    are attached as needed.
    Reasons:
    - avoids that a bunch of attributes could be None
    - selection score function is only called once instead of for every child node node
    - see at a glance which children are expanded (node.children is a dict/sparse array)
    """

    latent: Tensor
    belief: Optional[Tensor]

    value_sum: float
    visit_count: int
    reward: float
    value_pred: float

    probs: tuple[float, ...]
    children: MutableMapping[int, Node]

    def __init__(
        self,
        /,
        latent: Tensor,
        belief: Optional[Tensor],
        reward: float,
        value_pred: float,
        probs: Iterable[float],
    ) -> None:
        self.latent = latent
        self.belief = belief
        self.reward = reward
        self.value_pred = value_pred
        self.probs = tuple(probs)
        self.children = dict()
        self.visit_count = 0
        self.value_sum = 0

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def add_value(self, value: float) -> None:
        self.value_sum += value
        self.visit_count += 1

    def get_create_child(self, action: int, nets: Networks) -> Node:
        """
        Returns the child node for the given action, creating it first if necessary.
        """
        if action not in self.children:
            self.children[action] = self._create_child_at(action, nets)
        return self.children[action]

    @abstractmethod
    def _create_child_at(self, action: int, nets: Networks) -> Node:
        pass


class StateNode(Node):
    """
    Represents a normal game state. Child nodes are created using the dynamics network.
    """

    current_player: int

    def __init__(
        self,
        latent: Tensor,
        belief: Optional[Tensor],
        reward: float,
        nets: Networks,
    ):
        value_pred, probs, current_player = nets.prediction.si(latent, belief)
        self.current_player = cast(int, current_player.argmax().item())
        super().__init__(
            latent=latent,
            belief=belief,
            reward=reward,
            value_pred=value_pred.item(),
            probs=probs,
        )

    def _create_child_at(self, action: int, nets: Networks) -> Node:
        latent, belief, reward = nets.dynamics.si(
            self.latent,
            self.belief,
            F.one_hot(torch.tensor(action), C.game.instance.max_num_actions),
        )
        return StateNode(latent, belief, reward.item(), nets)


def ensure_visit_count(
    root: Node, visit_count: int, selection_fn: SelectionFn, nets: Networks
) -> None:
    """
    Run the tree search on a Node until the visit count is reached
    """
    for _ in range(visit_count - root.visit_count):
        node = root
        search_path = [root]
        was_expanded = True
        while was_expanded:
            action = selection_fn(node)
            was_expanded = action in node.children
            node = node.get_create_child(action, nets)
            search_path.append(node)

        # backpropagate
        r = node.value_pred
        for node in reversed(search_path):
            node.add_value(r)
            r = r * C.training.discount_factor + node.reward
