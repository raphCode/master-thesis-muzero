from typing import List, Optional

import torch

from game import PlayerType


class Node:
    """
    Represents a game state in the mcts search tree.
    A node is unexpanded after its creation, and can be expanded.
    That means some values of the node are calculated and for every possible action childs
    nodes are created.
    Some node attributes do not refer to the node itself, but to the action transition
    leading from the parent to this node.
    """

    parent: Optional[Node]
    value_sum: float
    visit_count: int
    reward: Optional[float]
    beliefs: Optional[torch.Tensor]
    player_type: Optional[PlayerType]
    latent_rep: Optional[torch.Tensor]
    children: List[Node]  # List index corresponds to action number

    # The following attributes refer to the transition from the parent to this node
    action: int
    prior: float
    reward: Optional[float]

    def __init__(self, parent: Optional[Node], action: int, prior: float):
        self.parent = parent
        self.action = action
        self.prior = prior
        self.children = []
        self.visit_count = 0
        self.value_sum = 0
        self.reward = None
        self.beliefs = None
        self.latent_rep = None
        self.player_type = None

    @property
    def is_expanded(self) -> bool:
        return len(self.children) > 0

    @property
    def value(self) -> float:
        return self.value_sum / self.visit_count

    @property
    def selection_score(self) -> float:
        return C.func.node_selection_score(self)

    def select_child(self) -> Node:
        """returns child node with highest selection_score"""
        assert self.player_type != PlayerType.Chance  # TODO: chance players
        score, node = max((node.selection_score, node) for node in self.children)
        return node

    def expand(self):
        """
        expands this node, that is:
        - populate our latent_rep, reward, node_type via network inference
        - add empty children for all possible actions
        """
        if self.parent is not None:
            # root node gets latent_rep and beliefs set externally
            tmp = C.nets.dynamics(parent.latent_rep, parent.beliefs, self.action)
            self.latent_rep, self.beliefs, self.reward = tmp
        value, probs, self.player_type = C.nets.prediction(self.latent_rep, self.beliefs)
        self.children = [Node(self, action, p) for action, p in enumerate(probs)]


def run_mcts(latent_rep: torch.Tensor, beliefs: torch.Tensor) -> Node:
    # this is called only for the player's own moves
    root = Node(None, None, None)
    root.latent_rep = latent_rep
    root.beliefs = beliefs
    root.reward = 0
    root.expand()
    for _ in range(C.param.mcts_num_simulations):
        node = root
        search_path = [root]
        while node.is_expanded:
            node = node.select_child()
        node.expand()

        # backpropagate
        # TODO: move this into a configurable function
        r = 0
        while node != root:
            # Accumulate reward predictions in the parents's value_sum
            r = r * C.param.discount + node.reward
            node = node.parent
            node.value_sum += r
            node.visit_count += 1

    return root
