from game import GameState
import pyspiel

class OpenSpielGameState(GameState):
    state: pyspiel.State
    invalid: bool

    def __init__(self, state: pyspiel.State):
        assert(state.is_player_node())
        self.state = state
        self.invalid = False

    @property
    def is_terminal(self) -> bool::
        return self.invalid or self.state.is_terminal()

    @property
    def current_player(self) -> int:
        return self.state.current_player()

    def apply_action(self, action: int):
        try:
            self.state.apply_action_with_legality_check(action)
            assert(self.state.is_player_node())
        except pyspiel.SpielError:
            self.invalid = True

def new_game(game: pyspiel.Game) -> OpenSpielGameState:
    """
    Start a new game and return the root state.
    Bind this with functool.partial to a Game created by pyspiel.load_game().
    """
    return OpenSpielGameState(game.new_initial_state())
