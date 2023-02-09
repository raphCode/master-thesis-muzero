from abc import ABC, abstractmethod
from typing import Unpack, Optional, TypeAlias, TypedDict
from collections.abc import Sequence

import torch
from attrs import frozen

from mcts import Node, ensure_visit_count
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
    target_policy: Sequence[float]
    mcts_value: float


RLBaseInitArgs: TypeAlias = tuple[Networks]


class RLBaseInitKwArgs(TypedDict):
    mcts_cfg: Optional[MctsConfig]


class RLBase(ABC):
    """
    Base class for reinforcement learning agents that use MCTS with neural nets.
    Throughout a game, they collect information that can be used to create training data.
    """

    nets: Networks
    mcts_cfg: MctsConfig
    root_node: Node
    representation: Observation | Latent

    def __init__(self, nets: Networks, mcts_cfg: Optional[MctsConfig] = None):
        self.nets = nets
        self.mcts_cfg = mcts_cfg or C.mcts
        self.reset_new_game()

    def reset_new_game(self) -> None:
        """
        Called whenever a new game starts.
        """
        self.representation = Latent(self.nets.initial_latent)
        self.root_node = Node(
            self.nets.initial_latent, self.nets.initial_belief, 0, self.nets
        )

    @abstractmethod
    def own_move(self, *observations: torch.Tensor) -> int:
        """
        Called when agent is at turn with current observations, returns action to take.
        Must set self.representation.
        """
        pass

    @abstractmethod
    def other_player_move(self, action: int) -> None:
        """
        Called when any other player made their move of the provided action.
        The information which action the other players took is intended be used only to
        create an accurate training trajectory, it must not be used for advantage in
        future own moves.
        Must set self.representation.
        """
        pass

    def create_training_info(self) -> TrainingInfo:
        """
        Called after each move, to record information for training.
        Uses self.representation so it is important this attribute is correcty set on own
        or other players' moves.
        """
        # This expands the part of the tree more that represents the actual trajectory.
        # This is an information leak when beliefs are searched in the tree in the next move.
        ensure_visit_count(
            self.root_node,
            self.mcts_cfg.iterations_value_estimate,
            self.mcts_cfg.node_selection_score_fn,
            self.nets,
        )
        return TrainingInfo(
            representation=self.representation,
            belief=self.root_node.belief,
            target_policy=self.mcts_cfg.node_target_policy_fn(self.root_node),
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
        latent, belief = self.nets.representation.si(None, *observations)
        self.root_node = Node(latent, belief, 0, self.nets)
        ensure_visit_count(
            self.root_node,
            self.mcts_cfg.iterations_move_selection,
            self.mcts_cfg.node_selection_score_fn,
            self.nets,
        )
        return self.mcts_cfg.node_action_fn(self.root_node)

    def other_player_move(self, action: int) -> None:
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
