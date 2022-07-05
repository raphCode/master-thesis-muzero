from abc import ABC, abstractmethod

import torch


class GameState(ABC):
    @property
    @abstractmethod
    def observation(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def rewards(self) -> tuple[float]:
        """Tuple of rewards, one for each player, starting at 0 for the first player"""
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
    def chance_outcomes(self) -> tuple[float]:
        pass

    @property
    @abstractmethod
    def legal_actions(self) -> int:
        pass

    @abstractmethod
    def apply_action(self, action: int):
        pass


class Game(ABC):
    @abstractmethod
    def new_initial_state(self) -> GameState:
        """Return a GameState which represents a new game at intial state"""
        pass

    @property
    @abstractmethod
    def num_players(self) -> int:
        """
        Number of players in the current playout from the last call to new_initial_state()
        Since this may keep state, a single Game instance should not be shared across
        multiple active playouts. In a multiprocess environment, this is less of a concern
        since by default, everything is copied anyways to the different processes.
        """
        pass

    @property
    @abstractmethod
    def observation_shapes(self) -> tuple[tuple[int]]:
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
        Used to set the size of the Action network in/outputs.
        """
        pass


class Player(ABC):
    @abstractmethod
    def request_action(self, state: GameState, game: Game) -> int:
        """Request an action from the player for the current game state"""
        pass
