import logging

import numpy as np
import torch
import torch.nn.functional as F

from mcts import Node, ensure_visit_count
from config import C
from globals import G
from rl_player import RLPlayer
from trajectory import (
    LatentInfo,
    PlayerType,
    ReplayBuffer,
    ObservationInfo,
    TrajectoryState,
)

rng = np.random.default_rng()
log = logging.getLogger(__name__)


def run_episode(replay_buffer: ReplayBuffer):
    players = C.player.instances
    rl_pids = {n for n, p in enumerate(players) if isinstance(p, RLPlayer)}

    mcts_nodes = {
        n: Node.from_latents(G.nets.initial_latent_rep, G.nets.initial_beliefs)
        for n in rl_pids
    }

    def get_and_update_mcts_tree(pid: int, action: int) -> Node:
        node = mcts_nodes[pid].get_action_subtree_and_prune_above(action)
        ensure_visit_count(node, C.mcts.iterations_value_estimate)
        mcts_nodes[pid] = node
        return node

    state = C.game.instance.new_initial_state()
    trajectories = {n: [] for n in rl_pids}

    for pid in rl_pids:
        players[pid].reset_new_game()

    for step in range(C.train.max_steps_per_episode):
        # Unsure about how to deal with non-terminal rewards or when exactly they occur
        assert state.is_chance or state.is_terminal or all(r == 0 for r in state.rewards)

        if state.is_terminal:
            break

        if state.is_chance:
            chance_outcomes = state.chance_outcomes
            action = rng.choice(C.game.instance.max_num_actions, p=chance_outcomes)
            state.apply_action(action)
            for tid, traj in trajectories.items():
                node = get_and_update_mcts_tree(tid, action)
                traj.append(
                    TrajectoryState(
                        info=LatentInfo(latent_rep=node.latent_rep, beliefs=node.beliefs),
                        player_type=PlayerType.Chance,
                        action=action,
                        target_policy=chance_outcomes,
                        mcts_value=node.value,
                        reward=C.game.calculate_reward(state.rewards, tid),
                    )
                )
            continue

        pid = state.current_player
        if pid in rl_pids:
            obs = state.observation
            action, old_beliefs, root_node = players[pid].request_action(obs)
            target_policy = C.mcts.node_target_policy_fn(root_node)
            mcts_nodes[pid] = root_node
        else:
            action = players[pid].request_action(state, C.game.instance)

        # estimate opponent behavior by averaging over their single moves:
        move_onehot = F.one_hot(torch.tensor(action), C.game.instance.max_num_actions)
        state.apply_action(action)

        for tid, traj in trajectories.items():
            if tid == pid:
                ts = TrajectoryState(
                    info=ObservationInfo(observation=obs, prev_beliefs=old_beliefs),
                    player_type=PlayerType.Self,
                    action=action,
                    target_policy=target_policy,
                    mcts_value=root_node.value,
                    reward=C.game.calculate_reward(state.rewards, pid),
                )
            else:
                node = get_and_update_mcts_tree(tid, action)
                ts = TrajectoryState(
                    info=LatentInfo(latent_rep=node.latent_rep, beliefs=node.beliefs),
                    player_type=PlayerType.Teammate
                    if C.player.is_teammate(pid, tid)
                    else PlayerType.Opponent,
                    action=action,
                    target_policy=move_onehot,
                    mcts_value=node.value,
                    reward=C.game.calculate_reward(state.rewards, tid),
                )
            traj.append(ts)

    for traj in trajectories.values():
        replay_buffer.add_trajectory(traj, game_terminated=state.is_terminal)

    log.info(f"Finished game ({step + 1} steps)")
