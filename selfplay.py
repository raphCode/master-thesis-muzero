from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
from attrs import frozen

from mcts import MCTS, TurnStatus
from config import C
from trajectory import TrajectoryState

if TYPE_CHECKING:
    import torch

    from util import ndarr_f32
    from networks import Networks
    from tensorboard_wrapper import TBStepLogger

rng = np.random.default_rng()

log = logging.getLogger(__name__)


@frozen
class SelfplayResult:
    moves: int
    game_completed: bool
    trajectory: list[TrajectoryState]


def run_episode(nets: Networks, tbs: TBStepLogger) -> SelfplayResult:
    state = C.game.instance.new_initial_state()
    traj = []
    n_players = C.game.instance.max_num_players
    scores = np.zeros(n_players)
    mcts = MCTS(nets, C.mcts)
    seen_obs = False

    def commit_step(
        action: int,
        *,
        obs: Optional[tuple[torch.Tensor, ...]],
        target_policy: ndarr_f32,
        mcts_value: Optional[ndarr_f32] = None,
    ) -> None:
        state.apply_action(action)

        def get_turn_status() -> int:
            if state.is_terminal:
                return TurnStatus.TERMINAL_STATE.target_index
            if state.is_chance:
                return TurnStatus.CHANCE_PLAYER.target_index
            return state.current_player_id

        nonlocal scores
        rewards = state.rewards
        scores += rewards

        if not seen_obs:
            # no point in recording a trajectory before the first observation
            return

        traj.append(
            TrajectoryState(
                observations=obs,
                turn_status=get_turn_status(),
                action=action,
                target_policy=target_policy,
                mcts_value=mcts_value
                if mcts_value is not None
                else np.zeros(n_players, dtype=np.float32),
                reward=rewards,
            )
        )

    def update_mcts(
        latent: torch.Tensor,
        chance_outcomes: Optional[ndarr_f32] = None,
    ) -> None:
        if chance_outcomes is not None:
            mcts.new_root(
                latent=latent,
                player_id=TurnStatus.CHANCE_PLAYER.target_index,
                valid_actions_mask=None,
                policy_override=chance_outcomes,
            )
        else:
            mcts.new_root(
                latent=latent,
                player_id=state.current_player_id,
                valid_actions_mask=state.valid_actions_mask,
            )
        mcts.ensure_visit_count(mcts.cfg.iterations)

    for n_step in range(C.training.max_steps_per_game):
        if state.is_terminal:
            n_step -= 1
            break

        if state.is_chance:
            chance_outcomes = state.chance_outcomes
            action = rng.choice(C.game.instance.max_num_actions, p=chance_outcomes)
            if seen_obs:
                update_mcts(mcts.get_latent_at(action), chance_outcomes)
            commit_step(
                action,
                obs=None,
                target_policy=chance_outcomes,
                mcts_value=None,
            )
            continue

        seen_obs = True
        obs = state.observation
        latent = nets.representation.si(*obs)
        update_mcts(latent)
        action = mcts.get_action()
        commit_step(
            action,
            obs=obs,
            target_policy=mcts.get_policy(),
            mcts_value=mcts.root.value,
        )

    tbs.add_scalar("selfplay/game length", n_step)
    trunc_msg = " (truncated)" * (not state.is_terminal)
    log.info(
        f"Finished selfplay game: {n_step + 1} steps{trunc_msg}, scores: "
        + " ".join(f"{s:.2f}" for s in scores)
    )
    for n, score in enumerate(scores):
        tbs.add_scalar(f"selfplay/score{n:02d}", score)

    return SelfplayResult(n_step, state.is_terminal, traj)
