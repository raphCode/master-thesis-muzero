from abc import ABC, abstractmethod
from typing import Unpack, Optional, TypeAlias, TypedDict

import torch
from attrs import frozen

from mcts import Node, StateNode, ensure_visit_count
from util import ndarr_f64
from config import C
from trajectory import Latent, Observation
from config.schema import MctsConfig
from networks.bases import Networks


@frozen(kw_only=True)
class TrainingInfo:
    """
    Information recorded by the RLPlayers to enable training.
    """

    representation: Observation | Latent
    belief: Optional[torch.Tensor]
    target_policy: ndarr_f64
    mcts_value: float


RLBaseInitArgs: TypeAlias = tuple[Networks]


class RLBaseInitKwArgs(TypedDict):
    mcts_cfg: Optional[MctsConfig]


class RLBase(ABC):
    """
    Base class for reinforcement learning agents that use MCTS with neural nets.
    Throughout a game, they collect information that can be used to create training data.

    The class keeps track of the game state by storing either the current Observation or a
    Latent representation. This state is used to return a TrainingInfo instance on
    create_training_info().
    The state is only ever advanced by calling advance_game_state(), not by own_move().
    The latter only stores the current observation.
    """

    nets: Networks
    mcts_cfg: MctsConfig
    root_node: Node
    representation: Observation | Latent

    def __init__(self, nets: Networks, mcts_cfg: Optional[MctsConfig] = None):
        self.nets = nets
        self.mcts_cfg = mcts_cfg or C.mcts

    def reset_new_game(self) -> None:
        """
        Called whenever a new game starts.
        """
        self.representation = Latent(self.nets.initial_latent)
        self.root_node = StateNode(
            self.nets.initial_latent, self.nets.initial_belief, 0, self.nets
        )

    @abstractmethod
    def own_move(self, *observations: torch.Tensor) -> int:
        """
        Called when agent is at turn with current observations, returns action to take.
        This must not advance the internal game state with the returned action, only store
        the observation.
        Must set self.representation to an Observation instance.
        """
        pass

    @abstractmethod
    def advance_game_state(self, action: int) -> None:
        """
        Called after any player (even this one) made their move of the provided action.
        The information which action the other players took is intended be used only to
        create an accurate training trajectory, it must not be used for advantage in
        future own moves.
        Must set self.representation to a Latent instance.
        """
        pass

    def create_training_info(self) -> TrainingInfo:
        """
        Called after each move, to record information for training.
        Uses self.representation so it is important this attribute is correcty set on own
        and other players' moves.
        """
        # Expand the part of the tree that represents the actual trajectory.
        # This may be an information leak if the next own move uses results from this tree
        ensure_visit_count(
            self.root_node,
            self.mcts_cfg.iterations_value_estimate,
            self.mcts_cfg.node_selection_score_fn,
            self.nets,
        )
        policy = self.mcts_cfg.node_target_policy_fn(self.root_node)
        assert len(policy) == C.game.instance.max_num_actions
        return TrainingInfo(
            representation=self.representation,
            belief=self.root_node.belief,
            target_policy=policy,
            mcts_value=self.root_node.value,
        )


class PerfectInformationRLPlayer(RLBase):
    """
    Uses beliefs to propagate information to new observations.
    To find the correct belief at the next own move, the tree search / dynamics network
    follows along the actual game trajectory with all actions of other players.
    This behaviour is only sound if the game is of perfect information, as any
    intermediate moves would be revealed in the next observation anyways.
    """

    def __init__(self, *args: Unpack[RLBaseInitArgs], **kwargs: Unpack[RLBaseInitKwArgs]):
        super().__init__(*args, **kwargs)

    def own_move(self, *observations: torch.Tensor) -> int:
        self.representation = Observation(observations)
        latent = self.nets.representation.si(*observations)
        self.root_node = StateNode(latent, self.root_node.belief, 0, self.nets)
        ensure_visit_count(
            self.root_node,
            self.mcts_cfg.iterations_move_selection,
            self.mcts_cfg.node_selection_score_fn,
            self.nets,
        )
        return self.mcts_cfg.node_action_fn(self.root_node)

    def advance_game_state(self, action: int) -> None:
        self.root_node = self.root_node.get_create_child(action, self.nets)
        self.representation = Latent(self.root_node.latent)


class NoBeliefsRLPlayer(PerfectInformationRLPlayer):
    """
    Like the original MuZero, uses no beliefs, runs new tree search on each observation.
    Disabling beliefs means the agent does not make use of intermediate actions in the
    next own move. This means the implementation is safe for imperfect-information games.
    """

    def __init__(self, *args: Unpack[RLBaseInitArgs], **kwargs: Unpack[RLBaseInitKwArgs]):
        super().__init__(*args, **kwargs)
        assert C.networks.belief_shape is None
