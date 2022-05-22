from abc import ABC, abstractmethod


class GameState(ABC):
    @property
    @abstractmethod
    def observation(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def rewards(self) -> Tuple[float]:
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
    def chance_outcomes(self) -> Tuple[float]:
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
    def max_num_players(self) -> int:
        """
        Maximum number of players that occur in playouts created by this game instance.
        Used to set the size of the PlayerOneHot network in/outputs.
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
