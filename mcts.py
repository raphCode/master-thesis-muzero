from abc import ABC, abstractmethod
from typing import Optional, cast
from collections.abc import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from config import C
from trajectory import PlayerType
from networks.bases import Networks

rng = np.random.default_rng()


class NodeBase(ABC):
    """
    A Node represents a game state in the monte carlo search tree.
    This class provides common functionality for different Node types.
    Child Nodes are never ObservationNodes, because in game states with observations a new
    search is started.

    Contrary to the MuZero original implementation, nodes are always expanded.
    Reasons:
    - avoids that a bunch of attributes could be None
    - selection score function is only called once instead of every child node
      node)
    - see at a glance which children are expanded (node.children is a dict/sparse array)
    """

    value_sum: float
    visit_count: int
    probs: tuple[float, ...]
    children: dict[int, "Node"]

    def __init__(self, probs: Iterable[float]) -> None:
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

    @abstractmethod
    def select_action(self) -> int:
        """
        Returns the action with the highest selection score.
        """
        pass

    def _chance_select(self) -> int:
        # Explicit dtype necessary since torch uses 32 and numpy 64 bits for floats by
        # default. The precision difference leads to the message 'probabilities to not
        # sum to 1' otherwise.
        return rng.choice(len(self.probs), p=np.array(self.probs, dtype=np.float32))

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


class ObservationNode(NodeBase):
    """
    Create child Nodes using the observation network, based on:
    - an observation
    - a random latent drawn from a list (usually from previous searches)

    Only used as the tree root.
    """

    latents: tuple[torch.Tensor, ...]

    def __init__(
        self,
        latents_with_probs: Optional[Iterable[tuple[torch.Tensor, float]]],
        *observations: torch.Tensor,
        nets: Networks,
    ):
        if latents_with_probs is not None:
            latent_tuple, probs = zip(*latents_with_probs)
            latents = torch.stack(latent_tuple)
        else:
            latents = None
            probs = (1,)

        # compute all the latents right away: it is very likely that all of them are
        # acessed at some time, because this is the tree root Node
        # single observation, multiple latents: might improve performance in some networks
        child_latents = nets.representation(
            latents, *(o.unsqueeze(0) for o in observations)
        )
        self.latents = tuple(child_latents)
        super().__init__(probs=probs)

    def select_action(self) -> int:
        return self._chance_select()

    def _create_child_at(self, action: int, nets: Networks) -> "Node":
        return Node(self.latents[action], 0.0, nets)


class Node(NodeBase):
    """
    Create child Nodes using the dynamics network, based on:
    - an action
    """

    reward: float
    value_pred: float
    latent: torch.Tensor
    player_type: PlayerType

    def __init__(self, latent: torch.Tensor, reward: float, nets: Networks):
        self.latent = latent
        self.reward = reward
        value_pred, probs, player_type = nets.prediction.si(self.latent)
        self.value_pred = value_pred.item()
        self.player_type = PlayerType(cast(int, player_type.argmax().item()))
        super().__init__(probs=probs)

    def select_action(self) -> int:
        if self.player_type == PlayerType.Chance:
            return self._chance_select()
        scores = C.mcts.node_selection_score_fn(self)
        return scores.index(max(scores))

    def _create_child_at(self, action: int, nets: Networks) -> "Node":
        latent, reward = nets.dynamics.si(
            self.latent,
            F.one_hot(torch.tensor(action), C.game.instance.max_num_actions),
        )
        return Node(latent, reward.item(), nets)


def run_mcts(latent: torch.Tensor, nets: Networks) -> Node:
    # this is called only for the player's own moves
    root = Node(latent, 0.0, nets)
    ensure_visit_count(root, C.mcts.iterations_move_selection, nets)
    return root


def ensure_visit_count(root: Node, visit_count: int, nets: Networks) -> None:
    """
    Run the tree search on a Node until the visit count is reached
    """
    for _ in range(visit_count - root.visit_count):
        node = root
        search_path = []
        was_expanded = True
        while was_expanded:
            search_path.append(node)
            action = node.select_action()
            was_expanded = action in node.children
            node = node.get_create_child(action, nets)

        # backpropagate
        r = node.value_pred
        for node in reversed(search_path):
            r = r * C.training.discount_factor + node.reward
            node.add_value(r)
