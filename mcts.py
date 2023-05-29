from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, cast
from collections.abc import Iterable, MutableMapping

import numpy as np
import torch
import torch.nn.functional as F

from config import C

if TYPE_CHECKING:
    from torch import Tensor

    from util import ndarr_f32, ndarr_f64
    from fn.selection import SelectionFn
    from networks.bases import Networks

log = logging.getLogger(__name__)


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

    probs: ndarr_f32 | ndarr_f64
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
        self.probs = np.asarray(probs)
        self.probs.flags.writeable = False
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
        latent, belief, reward, is_terminal = nets.dynamics.si(
            self.latent,
            self.belief,
            F.one_hot(torch.tensor(action), C.game.instance.max_num_actions),
        )
        if is_terminal.item() > 0.5 and C.training.loss_weights.terminal > 0:
            return TerminalNode(latent, belief, reward.item())
        return StateNode(latent, belief, reward.item(), nets)


class TerminalNode(Node):
    """
    Represents states beyond the predicted end of the game, with zero value prediction.
    Child nodes are always TerminalNodes with zero reward.

    By fixing value predictions and future rewards to zero, the accuracy of node values
    near the game end is improved, which enables better action selection and faster
    learning.
    This is subject to the condition that terminal states are correctly predicted.

    Compared to the alternative of predicting zero values and rewards for any state after
    the game end, learning the occurence of the game end directly provides these benefits:
    - more efficient in terms of time and compute
    - better generalisation for any number of states beyond the game end

    These predictions come from the dynamics rather than the prediction network because:
    - the 'terminalness' is not really associated with a game state, it is more related to
      a (state, action) tuple like a reward
    - a 'terminal score' can easily be learned in the training data of the the last game
      state without injecting extra training states

    The first TerminalNode in a tree search trajectory is important:
    - it contains the final reward
    - it stores a view count which might be relevant for deriving the target policy

    Children of this first TerminalNode are important as well; the tree search cannot
    simply be truncated: The terminal prediction might be wrong, and the latents / beliefs
    of some child nodes may still be needed for e.g. training data or some special
    RLPlayers.
    """

    def __init__(self, latent: Tensor, belief: Optional[Tensor], reward: float):
        super().__init__(
            latent=latent,
            belief=belief,
            reward=reward,
            value_pred=0.0,
            probs=np.full(
                C.game.instance.max_num_actions, 1 / C.game.instance.max_num_actions
            ),
        )

    def _create_child_at(self, action: int, nets: Networks) -> TerminalNode:
        latent, belief, _, _ = nets.dynamics.si(
            self.latent,
            self.belief,
            F.one_hot(torch.tensor(action), C.game.instance.max_num_actions),
        )
        return TerminalNode(latent, belief, 0)


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
        if len(search_path) - 1 > C.training.n_step_horizon:
            # The backpropagation currently calculates a node return value with all
            # rewards in the search path (n-step size only bounded by search depth).
            # Warn the user if the tree search goes deeper than the configured n-step
            # horizon size.
            # This is not a problem per se since the state values should converge to the
            # same values anyways, irrespectively of the n-step horizon size.
            # But sometimes small horizon sizes perform better empirically.
            log.warning(
                "tree search depth ({}) exceeds n-step horizon size ({})".format(
                    len(search_path) - 1, C.training.n_step_horizon
                )
            )

        r = node.value_pred
        for node in reversed(search_path):
            node.add_value(r)
            r = r * C.training.discount_factor + node.reward
