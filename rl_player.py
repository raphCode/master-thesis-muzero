from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Optional, TypeAlias, TypedDict

from attrs import frozen

from mcts import MCTS
from config import C
from networks import Networks
from games.bases import Player

if TYPE_CHECKING:
    from torch import Tensor

    from util import ndarr_f32
    from games.bases import GameState
    from config.schema import MctsConfig


@frozen(kw_only=True)
class TrainingInfo:
    """
    Information recorded by the RLPlayers to enable training.
    """

    observations: Optional[tuple[Tensor, ...]]
    target_policy: ndarr_f32
    mcts_value: ndarr_f32


RLBaseInitArgs: TypeAlias = tuple[Networks]


class RLBaseInitKwArgs(TypedDict):
    mcts_cfg: Optional[MctsConfig]


class RLBase(Player):
    """
    Base class for reinforcement learning agents that use MCTS with neural nets.
    Throughout a game, they collect information that can be used to create training data.

    The class keeps track of the game state by storing either the current Observation or a
    Latent representation. This state is used to return a TrainingInfo instance on
    create_training_info().
    The state is only ever advanced by calling advance_game_state(), not by
    request_action().
    The latter only stores the current observation.
    """

    nets: Networks
    mcts: MCTS
    observations: Optional[tuple[Tensor, ...]]

    def __init__(self, nets: Networks, mcts_cfg: Optional[MctsConfig] = None):
        self.nets = nets
        self.mcts = MCTS(nets, mcts_cfg or C.mcts)

    def reset_new_game(self, player_id: int) -> None:
        """
        Called whenever a new game starts.
        """
        self.mcts.reset_new_game(player_id)

    @abstractmethod
    def request_action(self, state: GameState) -> int:
        """
        Called when agent is at turn with current game state, returns action to take.
        This must not advance the internal game state with the returned action, only store
        the observation.
        Must set self.observations.
        """
        pass

    @abstractmethod
    def advance_game_state(self, action: int) -> None:
        """
        Called after any player (even this one) made their move of the provided action.
        The information which action the other players took is intended be used only to
        create an accurate training trajectory, it must not be used for advantage in
        future own moves.
        """
        pass

    def create_training_info(self) -> TrainingInfo:
        """
        Called after each move, to record information for training.
        Uses self.observations so it is important this attribute is correcty set on own
        and other players' moves.
        """
        # Expand the part of the tree that represents the actual trajectory.
        # This may be an information leak if the next own move uses results from this tree
        self.mcts.ensure_visit_count(self.mcts.cfg.iterations_value_estimate)
        return TrainingInfo(
            observations=self.observations,
            target_policy=self.mcts.get_policy(),
            mcts_value=self.mcts.root.value,
        )


class PerfectInformationRLPlayer(RLBase):
    """
    The tree search follows along the actual game trajectory with all actions of other
    players.
    This behaviour is only sound if the game is of perfect information, as any
    intermediate moves would be revealed in the next observation anyways.
    """

    def request_action(self, state: GameState) -> int:
        observations = state.observation
        self.observations = observations
        latent = self.nets.representation.si(*observations)
        self.mcts.new_root(latent, state.valid_actions_mask)
        self.mcts.ensure_visit_count(self.mcts.cfg.iterations_move_selection)
        return self.mcts.get_action()

    def advance_game_state(self, action: int) -> None:
        self.mcts.advance_root(action)
        self.observations = None
