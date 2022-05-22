from functools import cached_property

import pyspiel

from game import Game, GameState


class OpenSpielGameState(GameState):
    state: pyspiel.State
    invalid: bool
    _max_num_actions: int

    def __init__(self, state: pyspiel.State, max_num_actions: int):
        assert state.is_player_node()
        self.state = state
        self.invalid = False
        self._max_num_actions = max_num_actions

    @property
    def observation(self) -> torch.Tensor:
        return NotImplemented

    @property
    def rewards(self) -> Tuple[float]:
        return self.state.rewards()

    @property
    def is_terminal(self) -> bool:
        return self.invalid or self.state.is_terminal()

    @property
    def is_chance(self) -> bool:
        return self.state.is_chance_node()

    @property
    def current_player(self) -> int:
        return self.state.current_player()

    @abstractmethod
    def chance_outcomes(self) -> Tuple[float]:
        d = dict(self.state.chance_outcomes())
        return tuple(d.get(a, 0.0) for a in range(self._max_num_actions))

    @property
    def legal_actions(self) -> List[int]:
        return self.state.legal_actions()

    def apply_action(self, action: int):
        if self.invalid or action not in self.state.legal_actions():
            # also covers terminal states because the legal actions are empty then
            # TODO: set bad reward for offending player
            self.invalid = True
        else:
            # TODO: remove legality check, this is just a safety measure now
            self.state.apply_action_with_legality_check(action)
            assert self.state.is_player_node() or self.state.is_terminal()


class OpenSpielGame(Game):
    game: pyspiel.Game

    def __init__(self, game_name: str):
        self.game = pyspiel.load_game(game_name)

    def new_initial_state(self) -> OpenSpielGameState:
        return OpenSpielGameState(self.game.new_initial_state(), self.max_num_actions)

    @cached_property
    def num_players(self) -> int:
        # TODO: make variable number of players per playout possible
        return self.game.num_players()

    @cached_property
    def max_num_actions(self) -> int:
        return max(self.game.num_distinct_actions(), self.game.max_chance_outcomes())
