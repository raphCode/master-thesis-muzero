from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    import torch

    from util import ndarr_f32, ndarr_bool


class GameStateInitKwArgs(TypedDict):
    game: Game


class GameState(ABC):
    """
    A state in an active game / match. It keeps all the information necessary for all
    players and is mutated on moves.
    """

    game: Game

    def __init__(self, game: Game):
        self.game = game

    @property
    @abstractmethod
    def observation(self) -> tuple[torch.Tensor, ...]:
        """
        Return observation tensors. Multiple tensors are possible for different shapes.
        """
        pass

    @property
    @abstractmethod
    def valid_actions_mask(self) -> ndarr_bool:
        """
        Specifies wich actions are valid to take in the current state.
        Length of returned array should be game.max_num_actions
        """
        pass

    @property
    @abstractmethod
    def rewards(self) -> ndarr_f32:
        """
        Array of rewards, one for each player, index 0 for the first player
        """
        pass

    @property
    @abstractmethod
    def is_terminal(self) -> bool:
        pass

    @property
    @abstractmethod
    def is_chance(self) -> bool:
        pass

    @property
    @abstractmethod
    def current_player_id(self) -> int:
        """
        Return the zero-based index of the player currently at turn.
        This may raise an exception when called on chance or terminal nodes.
        """
        pass

    @property
    @abstractmethod
    def chance_outcomes(self) -> ndarr_f32:
        """
        On chance events, this can be called to get the action probabilities.
        Length of returned array should be game.max_num_actions
        """
        pass

    @abstractmethod
    def apply_action(self, action: int) -> None:
        """
        Mutate the game state and advance the game by making the given move.
        """
        pass


class Game(ABC):
    """
    Factory for game matches / playouts. Holds information valid across different matches.
    Matches may differ in player count or team groups, allowing the network to generalize
    across different player configurations.
    """

    @abstractmethod
    def new_initial_state(self) -> GameState:
        """
        Return a new GameState which represents a game / match at intial state
        """
        pass

    @property
    @abstractmethod
    def observation_shapes(self) -> tuple[tuple[int, ...], ...]:
        """
        Shape of observation tensors. Outer tuple is for defining multiple tensors.
        Used to set the size of the prediction network inputs.
        """
        pass

    @property
    @abstractmethod
    def max_num_actions(self) -> int:
        """
        Number of actions or chance outcomes, whichever is higher.
        Used to set the size of the dynamics network inputs.
        """
        pass

    @property
    @abstractmethod
    def max_num_players(self) -> int:
        """
        Maximum number of players in a match, excluding the chance player.
        Used to set the size of reward tuples, the team matrix and dynamics network
        outputs.
        """
        pass
