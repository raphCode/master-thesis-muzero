from abc import ABC, abstractmethod
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F

from config import C
from trajectory import PlayerType
from networks.bases import Networks

rng = np.random.default_rng()


class NodeBase(ABC):
    """
    Common functionality for different Node types.
    """

    value_sum: float
    visit_count: int
    children: dict[int, "Node"]  # list index corresponds to action number

    def __init__(self) -> None:
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

    @abstractmethod
    def select_action(self) -> int:
        """
        Returns the action with the highest selection score.
        """
        pass

    def get_create_child(self, action: int, nets: Networks) -> "Node":
        """
        Returns the child node for the given action, creating it first if necessary.
        """
        if action not in self.children:
            self.children[action] = self._create_child_at(action, nets)
        return self.children[action]

    @abstractmethod
    def _create_child_at(self, action: int, nets: Networks) -> "Node":
        pass


class Node(NodeBase):
    """
    Represents a game state in the monte carlo search tree.
    Contrary to the MuZero original implementation, nodes are always expanded.
    Reasons:
    - avoids that a bunch of attributes could be None
    - selection score function is only called once instead of every child node
      node)
    - see at a glance which children are expanded (node.children is a dict/sparse array)
    """

    reward: float
    value_pred: float
    latent: torch.Tensor
    player_type: PlayerType
    probs: tuple[float, ...]

    def __init__(self, latent: torch.Tensor, reward: float, nets: Networks):
        super().__init__()
        self.latent = latent
        self.reward = reward
        value_pred, probs, player_type = nets.prediction.si(self.latent)
        self.value_pred = value_pred.item()
        self.probs = tuple(probs)
        self.player_type = PlayerType(cast(int, player_type.argmax().item()))

    def select_action(self) -> int:
        if self.player_type == PlayerType.Chance:
            # Explicit dtype necessary since torch uses 32 and numpy 64 bits for floats by
            # default. The precision difference leads to the message 'probabilities to not
            # sum to 1' otherwise.
            return rng.choice(
                len(self.children), p=np.array(self.probs, dtype=np.float32)
            )
        scores = C.mcts.node_selection_score_fn(self)
        return scores.index(max(scores))

    def _create_child_at(self, action: int, nets: Networks) -> "Node":
        latent, reward = nets.dynamics.si(
            self.latent,
            F.one_hot(torch.tensor(action), C.game.instance.max_num_actions),
        )
        return Node(latent, reward.item(), nets)


def run_mcts(latent_rep: torch.Tensor, beliefs: torch.Tensor) -> Node:
    # this is called only for the player's own moves
    root = Node.from_latents(latent_rep, beliefs)
    root.expand()
    ensure_visit_count(root, C.mcts.iterations_move_selection)
    return root


def ensure_visit_count(root: Node, visit_count: int):
    """Run the tree search on an already expanded Node until the visit count is reached"""
    for _ in range(visit_count - root.visit_count):
        node = root
        while node.is_expanded:
            node = node.select_child()
        node.expand()

        # backpropagate
        # TODO: move this into a configurable function
        r = node.value_pred
        while node != root:
            # Accumulate reward predictions in the parents's value_sum
            r = r * C.train.discount_factor + node.reward
            node = node.parent
            node.value_sum += r
            node.visit_count += 1
