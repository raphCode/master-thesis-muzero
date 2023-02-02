from typing import Unpack, Optional
from functools import cached_property

import torch
import pyspiel

from .bases import Game, Teams, GameState, MatchData, GameStateInit


class OpenSpielGameState(GameState):
    state: pyspiel.State
    game: "OpenSpielGame"
    invalid: bool

    def __init__(self, state: pyspiel.State, **kwargs: Unpack[GameStateInit]):
        self.state = state
        self.invalid = False
        super().__init__(**kwargs)

    @property
    def observation(self) -> tuple[torch.Tensor]:
        return (
            torch.tensor(self.state.observation_tensor()).reshape(
                self.game.observation_shapes[0]
            ),
        )

    @property
    def rewards(self) -> tuple[float, ...]:
        if self.is_chance:
            return (0.0,) * self.game.max_num_players
        if self.invalid:
            assert self.game.bad_move_reward is not None
            tmp = [0.0] * self.game.max_num_players
            tmp[self.current_player_id] = self.game.bad_move_reward
            return tuple(tmp)
        return self.state.rewards()  # type: ignore [no-any-return]

    @property
    def is_terminal(self) -> bool:
        return self.invalid or self.state.is_terminal()

    @property
    def is_chance(self) -> bool:
        return self.state.is_chance_node()  # type: ignore [no-any-return]

    @property
    def current_player_id(self) -> int:
        return self.state.current_player()  # type: ignore [no-any-return]

    @property
    def chance_outcomes(self) -> tuple[float, ...]:
        d = dict(self.state.chance_outcomes())  # type: dict[int, float]
        return tuple(d.get(a, 0.0) for a in range(self.game.max_num_actions))

    def apply_action(self, action: int) -> None:
        assert not self.is_terminal
        if action not in self.state.legal_actions():
            if self.game.bad_move_action is not None:
                self.state.apply_action_with_legality_check(self.game.bad_move_action)
            else:
                self.invalid = True
        else:
            # TODO: remove legality check, this is just a safety measure now
            self.state.apply_action_with_legality_check(action)


class OpenSpielGame(Game):
    game: pyspiel.Game
    teams: Teams

    def __init__(
        self,
        game_name: str,
        bad_move_reward: Optional[float] = None,
        bad_move_action: Optional[int] = None,
        teams: list[list[int]] = [],
    ):
        self.game = pyspiel.load_game(game_name)
        self.bad_move_reward = bad_move_reward
        self.bad_move_action = bad_move_action
        assert (bad_move_reward is None) != (
            bad_move_action is None
        ), "Exactly one of 'bad_move_reward' or 'bad_move_action' must be given"
        self.teams = Teams(teams)
        assert self.game.observation_tensor_layout() == pyspiel.TensorLayout.CHW

    def new_initial_state(self) -> OpenSpielGameState:
        return OpenSpielGameState(
            self.game.new_initial_state(),
            game=self,
            match_data=MatchData(self.game.num_players(), self.teams),
        )

    @cached_property
    def max_num_players(self) -> int:
        return self.game.num_players()  # type: ignore [no-any-return]

    @cached_property
    def has_chance_player(self) -> bool:
        return True

    @cached_property
    def observation_shapes(self) -> tuple[tuple[int, ...]]:
        return (tuple(self.game.observation_tensor_shape()),)

    @cached_property
    def max_num_actions(self) -> int:
        return max(self.game.num_distinct_actions(), self.game.max_chance_outcomes())  # type: ignore [no-any-return]
