from __future__ import annotations

from typing import Any, Unpack, Optional, cast
from functools import cached_property

import numpy as np
import torch
import pyspiel  # type: ignore [import]
import torch.nn.functional as F

from util import ndarr_f32, ndarr_bool, optional_map

from .bases import Game, GameState, GameStateInitKwArgs


class OpenSpielGameState(GameState):
    state: pyspiel.State
    game: OpenSpielGame
    invalid: bool

    def __init__(self, state: pyspiel.State, **kwargs: Unpack[GameStateInitKwArgs]):
        self.state = state
        self.invalid = False
        super().__init__(**kwargs)

    @property
    def observation(self) -> tuple[torch.Tensor, torch.Tensor]:
        shapes = self.game.observation_shapes
        obs = torch.tensor(self.state.observation_tensor()).reshape(shapes[0])
        player_onehot = F.one_hot(
            torch.tensor(self.current_player_id), self.game.max_num_players
        )
        return obs, player_onehot

    @property
    def valid_actions_mask(self) -> ndarr_bool:
        if self.game.bad_move_reward is not None or self.game.bad_move_action is not None:
            return np.ones(self.game.max_num_actions, dtype=bool)
        mask = np.zeros(self.game.max_num_actions, dtype=bool)
        mask[list(self.state.legal_actions())] = True
        return mask

    @property
    def rewards(self) -> ndarr_f32:
        if self.is_chance:
            return np.zeros(self.game.max_num_players, dtype=np.float32)
        if self.invalid:
            assert self.game.bad_move_reward is not None
            ret = np.zeros(self.game.max_num_players, dtype=np.float32)
            ret[cast(int, self.state.current_player())] = self.game.bad_move_reward
            return ret
        return np.array(self.state.rewards())

    @property
    def is_terminal(self) -> bool:
        return self.invalid or self.state.is_terminal()

    @property
    def is_chance(self) -> bool:
        return cast(bool, self.state.is_chance_node())

    @property
    def current_player_id(self) -> int:
        assert not self.is_terminal
        assert not self.is_chance
        return cast(int, self.state.current_player())

    @property
    def chance_outcomes(self) -> ndarr_f32:
        d = dict(self.state.chance_outcomes())  # type: dict[int, float]
        probs = np.zeros(self.game.max_num_actions, dtype=np.float32)
        probs[list(d.keys())] = list(d.values())
        return probs

    def apply_action(self, action: int) -> None:
        assert not self.is_terminal
        if action not in self.state.legal_actions():
            if self.game.bad_move_action is not None:
                self.state.apply_action_with_legality_check(self.game.bad_move_action)
            else:
                assert (
                    self.game.bad_move_reward is not None
                ), "Illegal action and no bad move reward or action specified!"
                self.invalid = True
        else:
            self.state.apply_action(action)

    def __repr__(self) -> str:
        return repr(self.state)


class OpenSpielGame(Game):
    game: pyspiel.Game
    bad_move_reward: Optional[float]
    bad_move_action: Optional[int]

    def __init__(
        self,
        game_name: str,
        bad_move_reward: Optional[Any] = None,
        bad_move_action: Optional[Any] = None,
        **kwargs: dict[str, Any],
    ):
        self.game = pyspiel.load_game(game_name, kwargs)
        self.bad_move_reward = optional_map(float)(bad_move_reward)
        self.bad_move_action = optional_map(int)(bad_move_action)
        assert (
            bad_move_reward is None or bad_move_action is None
        ), "At most one of 'bad_move_reward' or 'bad_move_action' must be given"
        assert self.game.observation_tensor_layout() == pyspiel.TensorLayout.CHW
        t = self.game.get_type()
        assert (
            t.chance_mode != t.ChanceMode.SAMPLED_STOCHASTIC
        ), "Unsupported game: Sampled / implicit stochasticity!"
        assert (
            t.dynamics == t.Dynamics.SEQUENTIAL
        ), "Only sequential move games are supported!"
        assert (
            t.information == t.Information.PERFECT_INFORMATION
        ), "Only games with perfect information are supported!"

    def new_initial_state(self) -> OpenSpielGameState:
        return OpenSpielGameState(
            self.game.new_initial_state(),
            game=self,
        )

    @cached_property
    def max_num_players(self) -> int:
        return cast(int, self.game.num_players())

    @cached_property
    def observation_shapes(self) -> tuple[tuple[int, ...], tuple[int]]:
        shape = tuple(self.game.observation_tensor_shape())
        if len(shape) == 2:
            # add singleton channel dimension to enable use of convolution networks
            shape = (1, *shape)
        return (shape, (self.max_num_players,))

    @cached_property
    def max_num_actions(self) -> int:
        return max(
            cast(int, self.game.num_distinct_actions()),
            cast(int, self.game.max_chance_outcomes()),
        )
