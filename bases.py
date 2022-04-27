from abc import ABC, abstractmethod


class GameState(ABC):
    @property
    @abstractmethod
    def is_terminal(self) -> bool:
        pass

    @property
    @abstractmethod
    def current_player(self) -> int:
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
