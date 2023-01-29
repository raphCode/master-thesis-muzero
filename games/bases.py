from abc import ABC, abstractmethod

import torch


class GameState(ABC):
    """
    A state in an active game / match. It keeps all the information necessary for all
    players and is mutated on moves.
    """

    @property
    @abstractmethod
    def observation(self) -> tuple[torch.Tensor, ...]:
        """
        Return observation tensors. Multiple tensors are possible for different shapes.
        """
        pass

    @property
    @abstractmethod
    def rewards(self) -> tuple[float, ...]:
        """
        Tuple of rewards, one for each player, starting at 0 for the first player
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
    def current_player(self) -> int:
        pass

    @property
    @abstractmethod
    def chance_outcomes(self) -> tuple[float, ...]:
        """
        On chance events, this can be called to get the action probabilities.
        Length of returned tuple should be game.max_num_actions
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
        Used to set the size of reward tuples, the team matrix and prediction network
        outputs.
        """
        pass


class Player(ABC):
    @abstractmethod
    def request_action(self, state: GameState, game: Game) -> int:
        """Request an action from the player for the current game state"""
        pass
