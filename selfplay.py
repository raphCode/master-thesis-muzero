from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from collections.abc import Iterable, Sequence

import numpy as np
from attrs import frozen

from mcts import TurnStatus
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

    started_pids = set[int]()

    def commit_step(action: int) -> None:
        if state.is_chance:
            target_policy = state.chance_outcomes
            turn_status = TurnStatus.CHANCE_PLAYER.target_index
        else:
            target_policy = None
            turn_status = state.current_player_id

        state.apply_action(action)

        for player, pid, traj in zip(rlp.players, rlp.pids, rlp.trajectories):
            if pid not in started_pids:
                continue
            traj.append(
                TrajectoryState.from_training_info(
                    player.create_training_info(),
                    target_policy=target_policy,
                    turn_status=turn_status,
                    action=action,
                    reward=C.game.reward_fn(state, pid),
                )
            )
            player.advance_game_state(action)

    for n_step in range(C.training.max_steps_per_game):
        if state.is_terminal:
            n_step -= 1
            break

        if state.is_chance:
            chance_outcomes = state.chance_outcomes
            action = rng.choice(C.game.instance.max_num_actions, p=chance_outcomes)
            commit_step(action)
            continue

        curr_pid = state.current_player_id
        action = players[curr_pid].request_action(state)
        started_pids.add(curr_pid)

        commit_step(action)

    tbs.add_scalar("selfplay/game length", n_step)
    if isinstance(player_controller, SinglePC):
        reward = rlp.trajectories[0][-1].reward
        log.info(f"Finished selfplay game: length: {n_step}, reward: {reward:.2f}")
        tbs.add_scalar("selfplay/reward", reward)
    else:
        raise NotImplementedError()

    return SelfplayResult(n_step, state.is_terminal, rlp.trajectories)
