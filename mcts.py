import torch
from typing import List
from enum import IntEnum, auto

class NodeType(IntEnum):
    Chance = auto()
    OwnPlayer = auto()
    OtherPlayer = auto()

class Node:
    """
    Represents a game state in the mcts search tree.
    A node is unexpanded after its creation, and can be expanded.
    That means some values of the node are calculated and for every possible action childs
    nodes are created.
    Some node attributes do not refer to the node itself, but to the action transition
    leading from the parent to this node.
    """
    parent: Node
    value_sum: float
    visit_count: int
    node_type: NodeType
    latent_rep: Optional[torch.Tensor]
    children: List[Node] # List index corresponds to action number

    # The following attributes refer to the transition from the parent to this node
    action: int
    prior: float
    reward: float

    def __init__(self,  parent: Node, action: int, prior: float):
        self.parent = parent
        self.action = action
        self.prior = prior
        self.children = []

    @property
    def is_expanded(self) -> bool:
        return len(self.children) > 0

    @property
    def value(self) -> float:
        return self.value_sum / self.visit_count

    @property
    def selection_score(self) -> float:
        return C.funcs.node_selection_score(self)

    def select_child(self) -> Node:
        """returns child node with highest selection_score"""
        score, node = max((node.selection_score, node) for node in self.children)
        return node

    def expand(self):
        """
        expands this node, that is:
        - populate our latent_rep, reward, node_type via network inference
        - add empty children for all possible actions
        """
        if self.parent is not None:
            # root node gets latent_rep set externally
            self.reward, self.latent_rep = C.nets.dynamics(self.action, parent.latent_rep)
        value, probs, self.node_type = C.nets.prediction(self.latent_rep)
        self.children = [Node(self, action, p) for action, p in enumerate(probs)]
        self.children[action].latent_rep = latent_rep
        self.visit_count = 0
        self.value_sum = 0
