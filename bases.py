from abc import ABC, abstractmethod

class GameState(ABC):
    @abstractmethod
    def is_terminal(self) -> bool:
        pass

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

    @abstractmethod
    def num_players(self) -> int:
        pass

    @abstractmethod
    def num_actions(self) -> int:
        pass
