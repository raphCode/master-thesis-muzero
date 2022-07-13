from copy import deepcopy

import numpy as np
import torch

from mcts import Node, ensure_visit_count
from config import config as C
from rl_player import RLPlayer
from trajectory import PlayerType, ReplayBuffer, TrajectoryState

rng = np.random.default_rng()

replay_buffer = ReplayBuffer(C.train.replay_buffer_size)

players = C.player.agents
rl_pids = {n for n, p in enumerate(players) if isinstance(p, RLPlayer)}
initial_node = Node.from_latents(C.nets.initial_latent_rep, C.nets.initial_beliefs)

mcts_nodes = {n: deepcopy(initial_node) for n in rl_pids}


def get_update_mcts_tree(pid: int, action: int) -> Node:
    node = mcts_nodes[pid].get_action_subtree(action)
    ensure_visit_count(node, C.mcts.iterations_value_estimate)
    mcts_nodes[pid] = node
    return node


state = C.game.instance.new_initial_state()
trajectories = {n: [] for n in rl_pids}

for pid in rl_pids:
    players[pid].reset_new_game()

for _ in range(C.train.max_steps_per_episode):
    # Unsure about how to deal with non-terminal rewards or when exactly they occur
    assert all(r == 0 for r in state.rewards) or state.is_terminal

    if state.is_terminal:
        break

    if state.is_chance:
        chance_outcomes = state.chance_outcomes
        action = rng.choice(C.game.instance.max_num_actions, p=chance_outcomes)
        state.apply_action(action)
        for tid, traj in trajectories.items():
            node = get_update_mcts_tree(tid, action)
            traj.append(
                TrajectoryState(
                    observation=None,
                    latent_rep=node.latent_rep,
                    old_beliefs=players[tid].beliefs,
                    dyn_beliefs=node.beliefs,
                    player_type=PlayerType.Chance,
                    action=action,
                    target_policy=chance_outcomes,
                    value=node.value,
                    reward=C.game.calculate_reward(state.rewards, tid),
                )
            )
        continue

    pid = state.current_player
    player = players[pid]
    if pid in rl_pids:
        obs = state.observation
        action, old_beliefs, root_node = player.request_action(obs)
        target_policy = C.mcts.get_node_target_policy(root_node)
        mcts_nodes[pid] = root_node
    else:
        action = player.request_action(state, C.game.instance)

    # estimate opponent behavior by averaging over their single moves:
    move_onehot = F.one_hot(torch.tensor(action), C.game.instance.max_num_actions)
    state.apply_action(action)

    for tid, traj in trajectories.items():
        if tid == pid:
            ts = TrajectoryState(
                observation=obs,
                latent_rep=None,
                old_beliefs=old_beliefs,
                dyn_beliefs=None,
                player_type=PlayerType.Self,
                action=action,
                target_policy=target_policy,
                value=root_node.value,
                reward=C.game.calculate_reward(state.rewards, pid),
            )
        else:
            node = get_update_mcts_tree(tid, action)
            ts = TrajectoryState(
                observation=None,
                latent_rep=node.latent_rep,
                old_beliefs=players[tid].beliefs,
                dyn_beliefs=node.beliefs,
                player_type=PlayerType.Teammate
                if C.player.is_teammate(pid, tid)
                else PlayerType.Opponent,
                action=action,
                target_policy=move_onehot,
                value=node.value,
                reward=C.game.calculate_reward(state.rewards, tid),
            )
        traj.append(ts)

for traj in trajectories:
    replay_buffer.add_trajectory(traj)
