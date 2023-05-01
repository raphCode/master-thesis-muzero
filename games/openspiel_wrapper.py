from __future__ import annotations

from typing import Any, Unpack, Optional, cast
from functools import cached_property

import numpy as np
import torch
import pyspiel  # type: ignore

from util import ndarr_f64, optional_map

from .bases import Game, Teams, GameState, MatchData, GameStateInitKwArgs


class OpenSpielGameState(GameState):
    state: pyspiel.State
    game: OpenSpielGame
    invalid: bool

    def __init__(self, state: pyspiel.State, **kwargs: Unpack[GameStateInitKwArgs]):
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
        return cast(tuple[float, ...], self.state.rewards())

    @property
    def is_terminal(self) -> bool:
        return self.invalid or self.state.is_terminal()

    @property
    def is_chance(self) -> bool:
        return cast(bool, self.state.is_chance_node())

    @property
    def current_player_id(self) -> int:
        pid = self.state.current_player()
        if pid == pyspiel.PlayerId.CHANCE:
            assert self.game.chance_player_id is not None
            return self.game.chance_player_id
        return cast(int, pid)

    @property
    def chance_outcomes(self) -> ndarr_f64:
        d = dict(self.state.chance_outcomes())  # type: dict[int, float]
        probs = np.zeros(self.game.max_num_actions)
        probs[list(d.keys())] = list(d.values())
        return probs

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
    bad_move_reward: Optional[float]
    bad_move_action: Optional[int]

    def __init__(
        self,
        game_name: str,
        bad_move_reward: Optional[Any] = None,
        bad_move_action: Optional[Any] = None,
        teams: list[list[int]] = [],
    ):
        self.game = pyspiel.load_game(game_name)
        self.bad_move_reward = optional_map(float)(bad_move_reward)
        self.bad_move_action = optional_map(int)(bad_move_action)
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
        return cast(int, self.game.num_players())

    @cached_property
    def has_chance_player(self) -> bool:
        return True

    @cached_property
    def observation_shapes(self) -> tuple[tuple[int, ...]]:
        return (tuple(self.game.observation_tensor_shape()),)

    @cached_property
    def max_num_actions(self) -> int:
        return max(
            cast(int, self.game.num_distinct_actions()),
            cast(int, self.game.max_chance_outcomes()),
        )
