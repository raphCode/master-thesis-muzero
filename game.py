from abc import ABC, abstractmethod

CHANCE_PLAYER_ID = 0


class GameState(ABC):
    @property
    @abstractmethod
    def observation(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def rewards(self) -> List[float]:
        pass

    @property
    @abstractmethod
    def is_terminal(self) -> bool:
        pass

    @property
    @abstractmethod
    def current_player(self) -> int:
        pass

    @property
    @abstractmethod
    def chance_outcomes(self) -> List[float]:
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
        pass

    @property
    @abstractmethod
    def num_players(self) -> int:
        pass

    @property
    @abstractmethod
    def num_actions(self) -> int:
        pass


class Player(ABC):
    @abstractmethod
    def request_action(self, state: GameState, game: Game) -> int:
        """Request an action from the player for the current game state"""
        pass
