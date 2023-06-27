from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Self, Literal, Optional, TypeAlias, cast
from collections.abc import MutableMapping

import numpy as np
import torch
import torch.nn.functional as F

from config import C

if TYPE_CHECKING:
    from torch import Tensor

    from util import ndarr_f32, ndarr_f64, ndarr_bool
    from config.schema import MctsConfig
    from networks.bases import Networks

log = logging.getLogger(__name__)


class TurnStatus(Enum):
    CHANCE_PLAYER = 0  # the chance player is at turn
    TERMINAL_STATE = 1  # the game ended

    @classmethod
    def from_index(cls, index: int) -> int | Self:
        """
        Return the encoded TurnStatus or player id from the onehot vector.
        """
        n = C.game.instance.max_num_players
        if index < n:
            return index  # a normal zero-based player id
        return cls(index - n)  # a TurnStatus enum value

    @property
    def target_index(self) -> int:
        """
        Transform enum values to the corresponding onehot target index.
        """
        return C.game.instance.max_num_players + self.value


CurrentPlayer: TypeAlias = int | Literal[TurnStatus.CHANCE_PLAYER]


class Node(ABC):
    """
    A Node represents a game state in the monte carlo search tree.

    Contrary to the MuZero original implementation, nodes are always expanded and children
    are attached as needed.
    Reasons:
    - avoids that a bunch of attributes could be None
    - selection score function is only called once instead of for every child node
    - see at a glance which children are expanded (node.children is a dict/sparse array)
    """

    latent: Tensor
    belief: Tensor

    value_sum: float
    visit_count: int
    reward: float
    value_pred: float

    probs: ndarr_f32 | ndarr_f64
    children: MutableMapping[int, Node]

    mcts: MCTS

    def __init__(
        self,
        /,
        latent: Tensor,
        belief: Tensor,
        reward: float,
        value_pred: float,
        probs: ndarr_f32 | ndarr_f64,
        mcts: MCTS,
    ) -> None:
        self.latent = latent
        self.belief = belief
        self.reward = reward
        self.value_pred = value_pred
        self.probs = probs
        self.probs.flags.writeable = False
        self.children = dict()
        self.visit_count = 0
        self.value_sum = 0
        self.mcts = mcts

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def add_value(self, value: float) -> None:
        self.value_sum += value
        self.visit_count += 1

    def get_create_child(self, action: int) -> Node:
        """
        Returns the child node for the given action, creating it first if necessary.
        """
        if action not in self.children:
            self.children[action] = self._create_child_at(action)
        return self.children[action]

    @abstractmethod
    def _create_child_at(self, action: int) -> Node:
        pass


class StateNode(Node):
    """
    Represents a normal game state. Child nodes are created using the dynamics network.
    """

    player: CurrentPlayer
    mask: Optional[ndarr_bool]  # valid actions

    def __init__(
        self,
        latent: Tensor,
        belief: Tensor,
        reward: float,
        mcts: MCTS,
        /,
        current_player: Optional[CurrentPlayer] = None,
        valid_actions_mask: Optional[ndarr_bool] = None,
    ):
        self.mask = valid_actions_mask
        self.player = current_player or mcts.own_pid
        value_pred, policy = mcts.nets.prediction.si(latent, belief)
        probs = policy.detach().numpy()
        if valid_actions_mask is not None:
            # Just in case you want to disable the mask:
            # The correct place to do so is the game implementation. It must then return
            # an all-True mask and handle 'illegal' moves somehow.
            probs[~valid_actions_mask] = 0
            probs /= probs.sum()
        super().__init__(
            latent=latent,
            belief=belief,
            reward=reward,
            value_pred=value_pred.item(),
            probs=probs,
            mcts=mcts,
        )

    def _create_child_at(self, action: int) -> Node:
        latent, belief, reward, turn_onehot = self.mcts.nets.dynamics.si(
            self.latent,
            self.belief,
            F.one_hot(torch.tensor(action), C.game.instance.max_num_actions),
        )
        turn_status = TurnStatus.from_index(cast(int, turn_onehot.argmax().item()))
        if turn_status is TurnStatus.TERMINAL_STATE:
            return TerminalNode(latent, belief, reward.item(), self.mcts)
        return StateNode(
            latent,
            belief,
            reward.item(),
            self.mcts,
            current_player=turn_status,
        )


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

    def __init__(self, latent: Tensor, belief: Tensor, reward: float, mcts: MCTS):
        super().__init__(
            latent=latent,
            belief=belief,
            reward=reward,
            value_pred=0.0,
            probs=np.full(
                C.game.instance.max_num_actions, 1 / C.game.instance.max_num_actions
            ),
            mcts=mcts,
        )

    def _create_child_at(self, action: int) -> TerminalNode:
        latent, belief, _, _ = self.mcts.nets.dynamics.si(
            self.latent,
            self.belief,
            F.one_hot(torch.tensor(action), C.game.instance.max_num_actions),
        )
        return TerminalNode(latent, belief, 0, self.mcts)


class MCTS:
    """
    Stores and manages a search tree for an agent.
    """

    own_pid: int
    root: Node
    nets: Networks
    cfg: MctsConfig

    def __init__(
        self,
        nets: Networks,
        cfg: MctsConfig,
    ):
        self.nets = nets
        self.cfg = cfg

    def reset_new_game(self, player_id: int) -> None:
        self.own_pid = player_id
        self.new_root(
            self.nets.initial_latent,
            self.nets.initial_belief,
        )

    def new_root(
        self,
        latent: Tensor,
        belief: Tensor,
        valid_actions_mask: Optional[ndarr_bool] = None,
    ) -> None:
        self.root = StateNode(
            latent,
            belief,
            0,
            self,
            valid_actions_mask=valid_actions_mask,
        )

    def ensure_visit_count(self, count: int) -> None:
        """
        Runs the tree search on the root note until the visit count is reached.
        """
        for _ in range(count - self.root.visit_count):
            node = self.root
            search_path = [self.root]
            was_expanded = True
            while was_expanded:
                action = self.cfg.node_selection_score_fn(node)
                was_expanded = action in node.children
                node = node.get_create_child(action)
                search_path.append(node)

            # backpropagate
            if len(search_path) - 1 > C.training.n_step_horizon:
                # The backpropagation currently calculates a node return value with all
                # rewards in the search path (n-step size only bounded by search depth).
                # Warn the user if the tree search goes deeper than the configured n-step
                # horizon size.
                # This is not a problem per se since the state values should converge to
                # the same values anyways, irrespectively of the n-step horizon size.
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

    def get_action(self) -> int:
        return self.cfg.node_action_fn(self.root)

    def get_policy(self) -> ndarr_f64:
        policy = self.cfg.node_target_policy_fn(self.root)
        assert len(policy) == C.game.instance.max_num_actions
        return policy

    def advance_root(self, action: int) -> None:
        """
        Replace the root node with its child of the given action.
        """
        self.root = self.root.get_create_child(action)
