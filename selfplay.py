from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from collections.abc import Iterable, Sequence

import numpy as np
from attrs import define, frozen

from mcts import TurnStatus
from util import RingBuffer, TimeProfiler
from config import C
from rl_player import RLBase
from trajectory import TrajectoryState
from player_controller import PCBase, SelfplayPC
from games.openspiel_wrapper import OpenSpielGameState

if TYPE_CHECKING:
    from games.bases import Player
    from tensorboard_wrapper import TBStepLogger

rng = np.random.default_rng()

log = logging.getLogger(__name__)


@define
class Counter:
    games: int
    moves_current_game: int
    moves_total: int


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


timer = TimeProfiler()


def run_episode(player_controller: PCBase, tbs: TBStepLogger) -> SelfplayResult:
    debug_text_buffer = RingBuffer[str](3)

    state = C.game.instance.new_initial_state()
    players = player_controller.get_players(state.match_data)
    assert len(players) == state.match_data.num_players
    rlp = RLPlayers(players)

    for pid, player in zip(rlp.pids, rlp.players):
        player.reset_new_game(pid)

    scores = [0.0 for _ in rlp.players]

    for n_move in range(C.game.max_steps_per_game):
        if state.is_chance:
            chance_outcomes = state.chance_outcomes
            action = rng.choice(C.game.instance.max_num_actions, p=chance_outcomes)
            state.apply_action(action)
            for player, pid, traj in zip(rlp.players, rlp.pids, rlp.trajectories):
                rew = C.game.reward_fn(state, pid)
                scores[pid] += rew
                traj.append(
                    TrajectoryState.from_training_info(
                        player.create_training_info(),
                        target_policy=chance_outcomes,
                        turn_status=TurnStatus.CHANCE_PLAYER.target_index,
                        action=action,
                        reward=rew,
                    )
                )
                player.advance_game_state(action)
            continue

        curr_pid = state.current_player_id
        curr_player = players[curr_pid]
        # with timer:
        # action = curr_player.request_action(state)
        action = curr_player.request_action(state)

        def debug_text() -> str:
            def wide_text(text: str) -> str:
                return "\n".join(" ".join(line) for line in text.splitlines())

            assert isinstance(curr_player, RLBase)
            assert isinstance(state, OpenSpielGameState)
            return "\n".join(
                [
                    # wide_text(state.state.observation_string()),
                    repr(state),
                    curr_player.mcts.debug_dump_tree(maxdepth=5),
                ]
            )

        # print(debug_text())
        print(repr(state))
        debug_text_buffer.append(debug_text())

        state.apply_action(action)

        for player, pid, traj in zip(rlp.players, rlp.pids, rlp.trajectories):
            rew = C.game.reward_fn(state, pid)
            scores[pid] += rew
            traj.append(
                TrajectoryState.from_training_info(
                    player.create_training_info(),
                    turn_status=curr_pid,
                    action=action,
                    reward=rew,
                )
            )
            player.advance_game_state(action)

        if state.is_terminal:
            break

    print("=" * 50)
    # print(debug_text_buffer[-3])
    # print(debug_text_buffer[-2])
    # print(debug_text_buffer[-1])
    # print(debug_text_buffer[0])
    for txt in debug_text_buffer:
        print(txt)

    tbs.add_scalar("selfplay/game length", n_move)
    if isinstance(player_controller, SelfplayPC):
        trunc_msg = " (truncated)" * (not state.is_terminal)
        log.info(
            f"Finished selfplay game: {n_move+1} moves{trunc_msg}, scores: "
            + " ".join(f"{s:.2f}" for s in scores)
        )
        for n, score in enumerate(scores):
            tbs.add_scalar(f"selfplay/score{n:02d}", score)
    else:
        raise NotImplementedError()

    return SelfplayResult(n_move, state.is_terminal, rlp.trajectories)
