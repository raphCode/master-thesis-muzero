from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from collections.abc import Iterable, Sequence

import numpy as np
from attrs import frozen

from config import C
from rl_player import RLBase
from trajectory import TrajectoryState
from player_controller import PCBase, SinglePC

if TYPE_CHECKING:
    from games.bases import Player
    from tensorboard_wrapper import TBStepLogger

rng = np.random.default_rng()

log = logging.getLogger(__name__)


@frozen
class SelfplayResult:
    moves: int
    game_completed: bool
    trajectories: Sequence[list[TrajectoryState]]


class RLPlayers:
    """
    Small helper class to bundle RLPlayers instances, their player ids and trajectories.
    """

    pids: tuple[int, ...]
    players: tuple[RLBase, ...]
    trajectories: tuple[list[TrajectoryState], ...]

    def __init__(self, all_players: Iterable[RLBase | Player]):
        self.pids, self.players = zip(
            *((pid, p) for pid, p in enumerate(all_players) if isinstance(p, RLBase))
        )
        self.trajectories = tuple([] for _ in self.players)


def run_episode(player_controller: PCBase, tbs: TBStepLogger) -> SelfplayResult:
    state = C.game.instance.new_initial_state()
    players = player_controller.get_players(state.match_data)
    assert len(players) == state.match_data.num_players
    rlp = RLPlayers(players)

    for pid, player in zip(rlp.pids, rlp.players):
        player.reset_new_game(pid)

    for n_move in range(C.training.max_moves_per_game):
        if state.is_terminal:
            break

        if state.current_player_id == C.game.instance.chance_player_id:
            chance_outcomes = state.chance_outcomes
            action = rng.choice(C.game.instance.max_num_actions, p=chance_outcomes)
            state.apply_action(action)
            for player, pid, traj in zip(rlp.players, rlp.pids, rlp.trajectories):
                traj.append(
                    TrajectoryState.from_training_info(
                        player.create_training_info(),
                        target_policy=chance_outcomes,
                        current_player=C.game.instance.chance_player_id,
                        action=action,
                        reward=C.game.reward_fn(state, pid),
                    )
                )
                player.advance_game_state(action)
            continue

        curr_pid = state.current_player_id
        curr_player = players[curr_pid]
        if isinstance(curr_player, RLBase):
            action = curr_player.own_move(*state.observation)
        else:
            action = curr_player.request_action(state, C.game.instance)

        state.apply_action(action)

        for player, pid, traj in zip(rlp.players, rlp.pids, rlp.trajectories):
            traj.append(
                TrajectoryState.from_training_info(
                    player.create_training_info(),
                    current_player=curr_pid,
                    action=action,
                    reward=C.game.reward_fn(state, pid),
                )
            )
            player.advance_game_state(action)

    tbs.add_scalar("selfplay/game length", n_move)
    if isinstance(player_controller, SinglePC):
        reward = rlp.trajectories[0][-1].reward
        log.info(f"Finished selfplay game: length: {n_move}, reward: {reward:.2f}")
        tbs.add_scalar("selfplay/reward", reward)
    else:
        raise NotImplementedError()

    return SelfplayResult(n_move, state.is_terminal, rlp.trajectories)
