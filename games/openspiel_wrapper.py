import pyspiel

from game import Game, GameState


class OpenSpielGameState(GameState):
    state: pyspiel.State
    invalid: bool

    def __init__(self, state: pyspiel.State):
        assert state.is_player_node()
        self.state = state
        self.invalid = False

    @property
    def observation(self) -> torch.Tensor:
        return NotImplemented

    @property
    def rewards(self) -> List[float]:
        return self.state.rewards()

    @property
    def is_terminal(self) -> bool:
        return self.invalid or self.state.is_terminal()

    @property
    def current_player(self) -> int:
        # Add one because chance player is originally id -1
        return self.state.current_player() + 1

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
        return self.game.new_initial_state()

    @property
    def num_players(self) -> int:
        return self.game.num_players()

    @property
    def max_num_actions(self) -> int:
        return max(self.game.num_distinct_actions(), self.game.max_chance_outcomes())
