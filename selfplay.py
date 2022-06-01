from copy import deepcopy

import numpy as np
import torch

from game import CHANCE_PLAYER_ID
from mcts import Node, ensure_visit_count
from config import config as C
from rl_player import RLPlayer
from trajectory import PlayerType, ReplayBuffer, TrajectoryState

rng = np.random.default_rng()

players = ...
rl_pids = {n for n, p in players.items() if isinstance(p, RLPlayer)}
replay_buffer = ReplayBuffer(C.param.replay_buffer_size)

state = C.game.new_initial_state()
trajectories = {n: [] for n in rl_pids}

initial_node = Node.from_latents(C.nets.initial_latent_rep, C.nets.initial_beliefs)
mcts_nodes = {n: deepcopy(initial_node) for n in rl_pids}


def get_update_mcts_tree(pid: int, action: int) -> Node:
    node = mcts_nodes[pid].get_action_subtree(action)
    ensure_visit_count(node, C.param.mcts_iter_value_estimate)
    mcts_nodes[rid] = node
    return node


for _ in range(C.param.max_steps_per_episode):
    # Unsure about how to deal with non-terminal rewards or when exactly they occur
    assert all(r == 0 for r in state.rewards) or state.is_terminal

    if state.is_terminal:
        break

    if state.is_chance:
        chance_outcomes = state.chance_outcomes
        action = rng.choice(C.game.max_num_actions, p=chance_outcomes)
        state.apply_action(action)
        for tid, traj in trajectories.items():
            node = get_update_mcts_tree(tid, action)
            traj.append(
                TrajectoryState(
                    observation=None,
                    latent_rep=node.latent_rep,
                    old_beliefs=player[tid].beliefs,
                    dyn_beliefs=node.beliefs,
                    player_type=PlayerType.Chance,
                    action=action,
                    target_policy=chance_outcomes,
                    value=node.value,
                    reward=C.func.calculate_reward(state.rewards, tid),
                )
            )
        continue

    pid = state.current_player
    player = players[pid]
    if pid in rl_pids:
        obs = state.observation
        action, old_beliefs, root_node = player.request_action(obs)
        target_policy = C.func.mcts_root2target_policy(root_node)
        mcts_nodes[pid] = root_node
    else:
        action = player.request_action(state, C.game)

    # estimate opponent behavior by averaging over their single moves:
    move_onehot = F.one_hot(torch.tensor(action), C.game.max_num_actions)
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
                reward=C.func.calculate_reward(state.rewards, pid),
            )
        else:
            node = get_update_mcts_tree(tid, action)
            ts = TrajectoryState(
                observation=None,
                latent_rep=node.latent_rep,
                old_beliefs=player[tid].beliefs,
                dyn_beliefs=node.beliefs,
                player_type=PlayerType.Teammate
                if C.func.is_teammate(pid, tid)
                else PlayerType.Opponent,
                action=action,
                target_policy=move_onehot,
                value=node.value,
                reward=C.func.calculate_reward(state.rewards, tid),
            )
        traj.append(ts)

for traj in trajectories:
    replay_buffer.add_trajectory(traj)
