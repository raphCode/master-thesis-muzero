from functools import cached_property

import torch
import pyspiel

from .bases import Game, GameState


class OpenSpielGameState(GameState):
    state: pyspiel.State
    game: "OpenSpielGame"
    invalid: bool

    def __init__(self, state: pyspiel.State, game: "OpenSpielGame"):
        self.state = state
        self.game = game
        self.invalid = False

    @property
    def observation(self) -> tuple[torch.Tensor]:
        return (
            torch.tensor(self.state.observation_tensor()).reshape(
                self.game.observation_shapes[0]
            ),
        )

    @property
    def rewards(self) -> tuple[float]:
        if self.is_chance:
            return (0,) * self.game.num_players
        if self.invalid:
            tmp = [0] * self.game.num_players
            tmp[self.current_player] = self.game.bad_move_reward
            return tuple(tmp)
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

    @property
    def chance_outcomes(self) -> tuple[float]:
        d = dict(self.state.chance_outcomes())
        return tuple(d.get(a, 0.0) for a in range(self.game.max_num_actions))

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

    def __init__(self, game_name: str, bad_move_reward: float):
        self.game = pyspiel.load_game(game_name)
        self.bad_move_reward = bad_move_reward
        assert self.game.observation_tensor_layout() == pyspiel.TensorLayout.CHW

    def new_initial_state(self) -> OpenSpielGameState:
        return OpenSpielGameState(
            self.game.new_initial_state(),
            self,
        )

    @cached_property
    def num_players(self) -> int:
        return self.game.num_players()

    @cached_property
    def observation_shapes(self) -> tuple[tuple[int]]:
        return (self.game.observation_tensor_shape(),)

    @cached_property
    def max_num_actions(self) -> int:
        return max(self.game.num_distinct_actions(), self.game.max_chance_outcomes())
