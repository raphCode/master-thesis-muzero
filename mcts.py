from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from config import C
from globals import G
from trajectory import PlayerType

rng = np.random.default_rng()


class Node:
    """
    Represents a game state in the mcts search tree.
    A node is unexpanded after its creation, and can be expanded.
    That means some values of the node are calculated and for every possible action childs
    nodes are created.
    Some node attributes do not refer to the node itself, but to the action transition
    leading from the parent to this node.
    """

    parent: "Node"
    value_sum: float
    visit_count: int
    reward: Optional[float]
    value_pred: Optional[float]
    beliefs: Optional[torch.Tensor]
    player_type: Optional[PlayerType]
    latent_rep: Optional[torch.Tensor]
    children: list["Node"]  # list index corresponds to action number

    # The following attributes refer to the transition from the parent to this node
    action: int
    prior: float
    reward: Optional[float]

    def __init__(self, parent: "Node", action: int, prior: float):
        self.parent = parent
        self.action = action
        self.prior = prior
        self.children = []
        self.visit_count = 0
        self.value_sum = 0
        self.reward = None
        self.beliefs = None
        self.value_pred = None
        self.latent_rep = None
        self.player_type = None

    @classmethod
    def from_latents(cls, latent_rep: torch.Tensor, beliefs: torch.Tensor) -> "Node":
        """Construct a Node from latent_rep and beliefs, suitable as a new tree root"""
        self = cls(None, None, None)
        self.latent_rep = latent_rep
        self.beliefs = beliefs
        self.reward = 0
        return self

    @property
    def is_expanded(self) -> bool:
        return len(self.children) > 0

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def select_child(self) -> "Node":
        """returns child node with highest selection_score"""
        if self.player_type == PlayerType.Chance:
            # Explicit dtype necessary since torch uses 32 and numpy 64 bits for floats by
            # default. The precision difference leads to the message 'probabilities to not
            # sum to 1' otherwise.
            probs = np.array([c.prior for c in self.children], dtype=np.float32)
            return rng.choice(self.children, p=probs)
        return max(self.children, key=C.mcts.get_node_selection_score)

    def expand(self):
        """
        expands this node, that is:
        - populate our latent_rep, reward, node_type via network inference
        - add empty children for all possible actions
        """
        if self.parent is not None:
            # root node gets latent_rep and beliefs set externally
            self.latent_rep, self.beliefs, reward = G.nets.dynamics.si(
                self.parent.latent_rep,
                self.parent.beliefs,
                F.one_hot(torch.tensor(self.action), C.game.instance.max_num_actions),
            )
            self.reward = reward.item()
        value_pred, probs, player_type = G.nets.prediction.si(
            self.latent_rep, self.beliefs
        )
        self.value_pred = value_pred.item()
        self.player_type = PlayerType(player_type.argmax().item())
        self.children = [Node(self, action, p) for action, p in enumerate(probs.tolist())]

    def ensure_expanded(self):
        if not self.is_expanded:
            self.expand()

    def get_action_subtree_and_prune_above(self, action: int) -> "Node":
        """
        Returns the child with the given action as a new tree root, discards parent tree.
        The rest of the tree above the returned node is effectively broken after this
        operation and should not be used anymore.
        All data in the subtree below the returned node is retained and can still be used.
        """
        self.ensure_expanded()
        node = self.children[action]
        node.ensure_expanded()
        node.parent = None
        node.action = None
        node.reward = 0
        return node


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
