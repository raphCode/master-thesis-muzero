from __future__ import annotations

import logging
import itertools
from typing import Any, Unpack, Optional, TypeAlias
from functools import cached_property
from collections.abc import Generator

import numpy as np
import torch
import torch.nn.functional as F

from util import ndarr_f32, ndarr_bool

from .bases import Game, GameState, GameStateInitKwArgs
from .carchess import Map

StateGen: TypeAlias = Generator[
    tuple[int, float, ndarr_bool, str] | tuple[None, float, int, str], int, None
]
log = logging.getLogger(__name__)


class CarchessGameState(GameState):
    game: CarchessGame
    map: Map
    round: int
    sm: StateGen

    curr_player: Optional[int]
    action_mask: ndarr_bool
    reward: float
    spawn_capacity: Optional[int]
    info: str
    score: float

    def __init__(self, m: Map, **kwargs: Unpack[GameStateInitKwArgs]):
        def state_machine() -> StateGen:
            for self.round in itertools.count(0):
                action_mask = self.map.tl.lights.flatten()
                for n in range(self.game.max_num_players):
                    action = yield n, 0, action_mask, f"Player {n} decision"
                    self.map.tl_flat_action(action)
                    action_mask[action] = False

                goal, crash = 0, 0
                reward = 0.0
                for n in range(self.game.simulation_steps):
                    g, c = self.map.simulation_step()
                    goal += g
                    crash += c

                reward = goal * self.game.reward_goal + crash * self.game.reward_crash
                info = (
                    f"Simulation result ({self.game.simulation_steps} steps): "
                    f"{g} goal, {c} crash, reward: {reward}\n"
                )
                for n, l in enumerate(self.map.layers):
                    chance_action = (
                        yield None,
                        reward,
                        l.remaining_spawn_capacity(self.game.max_density),
                        info + f"Spawn count update layer {n}",
                    )
                    info = ""
                    reward = 0
                    random_number = self.game.min_spawn + chance_action
                    self.map.update_spawn_count_explicit(
                        n, random_number, self.game.max_density
                    )

        super().__init__(**kwargs)
        self.map = m
        self.score = 0
        self.sm = state_machine()
        self._update_state()

    def _update_state(self, action: Optional[int] = None) -> None:
        self.curr_player, self.reward, data, self.info = self.sm.send(action)  # type: ignore [arg-type] # noqa: E501
        self.info = f"Last action: {action}\n" + self.info
        if self.curr_player is None:
            # chance node
            assert isinstance(data, int)
            self.spawn_capacity = data
        else:
            self.spawn_capacity = None
            assert isinstance(data, np.ndarray)
            self.action_mask = data

    @property
    def observation(self) -> tuple[torch.Tensor, ...]:
        obs = np.concatenate(
            [
                self.map.car_observation(),
                np.expand_dims(self.map.tl_observation(), axis=0),
                self.map.spawn_count_observation(self.game.max_density),
            ]
        )
        return (
            torch.tensor(obs, dtype=torch.float32),
            F.one_hot(torch.tensor(self.current_player_id), self.game.max_num_players),
            F.one_hot(torch.tensor(self.round), self.game.rounds),
        )

    @property
    def valid_actions_mask(self) -> ndarr_bool:
        assert not self.is_chance
        return self.action_mask

    @property
    def rewards(self) -> ndarr_f32:
        return np.full(self.game.max_num_players, self.reward, dtype=np.float32)

    @property
    def is_terminal(self) -> bool:
        return self.round >= self.game.rounds

    @property
    def is_chance(self) -> bool:
        assert (self.spawn_capacity is not None) == (self.curr_player is None)
        return self.spawn_capacity is not None

    @property
    def current_player_id(self) -> int:
        assert self.curr_player is not None
        return self.curr_player

    @property
    def chance_outcomes(self) -> ndarr_f32:
        assert self.spawn_capacity is not None
        b = np.arange(self.game.min_spawn, self.game.max_spawn + 1)
        b = np.minimum(self.spawn_capacity, b)
        counts = np.bincount(b).astype(np.float32)
        counts /= counts.sum()
        counts.resize(self.game.max_num_actions, refcheck=False)
        return counts

    def apply_action(self, action: int) -> None:
        if self.spawn_capacity is not None:
            assert action <= self.spawn_capacity
        else:
            assert self.action_mask[action]
        self._update_state(action)
        self.score += self.reward

    def __repr__(self) -> str:
        return (
            f"Round: {self.round + 1} Score: {self.score}\n"
            f"{self.info}\n" + self.map.ascii_art()
        )


class CarchessGame(Game):
    map: Map

    def __init__(
        self,
        map_name: Any = "tutorial",
        min_spawn: int = 0,
        max_spawn: int = 4,
        max_density: float = 0.5,
        num_players: int = 2,
        rounds: int = 10,
        simulation_steps: int = 5,
        reward_goal: float = 2,
        reward_crash: float = -1,
    ) -> None:
        self.min_spawn = int(min_spawn)
        self.max_spawn = int(max_spawn)
        self.max_density = float(max_density)
        self.num_players = int(num_players)
        self.rounds = int(rounds)
        self.simulation_steps = int(simulation_steps)
        self.reward_goal = float(reward_goal)
        self.reward_crash = float(reward_crash)
        self.map = Map(map_name)
        log.info(
            f"Initialized carchess game with map '{map_name}'\n"
            "Estimated state space complexity:\n"
            + str(self.map.estimate_state_space_complexity(self.max_density, self.rounds))
        )

    def new_initial_state(self) -> GameState:
        self.map.reset(prepopulate=0)
        self.map.update_spawn_counts_random(
            self.min_spawn, self.max_spawn, self.max_density
        )
        return CarchessGameState(self.map, game=self)

    @cached_property
    def observation_shapes(self) -> tuple[tuple[int, ...], ...]:
        # obs = CarchessGameState(self.map, game=self).observation
        # return tuple(tuple(o.shape for o in obs)
        return (
            (
                self.max_spawn_count  # onehot spawn maps
                + 1  # for zero spawn count map
                + 1  # traffic light map
                + len(self.map.layers),  # one car map per layer
                *self.map.size,
            ),
            (self.max_num_players,),
            (self.rounds,),
        )

    @cached_property
    def max_num_actions(self) -> int:
        w, h = self.map.size
        return max(w * h, self.max_spawn)

    @cached_property
    def max_num_players(self) -> int:
        return self.num_players

    @cached_property
    def max_spawn_count(self) -> int:
        return int(self.max_density * self.map.max_lane_capacity)
