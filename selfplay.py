import logging
import contextlib
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from mcts import Node, ensure_visit_count
from config import config as C
from rl_player import RLPlayer
from trajectory import PlayerType, ReplayBuffer, TrajectoryState

from torch.utils.tensorboard import SummaryWriter
rng = np.random.default_rng()
log = logging.getLogger(__name__)
from typing import Callable
import matplotlib.pyplot as plt


def run_episode(replay_buffer: ReplayBuffer, sw: SummaryWriter, n: int):
    players = C.player.instances
    rl_pids = {n for n, p in enumerate(players) if isinstance(p, RLPlayer)}
    initial_node = Node.from_latents(C.nets.initial_latent_rep, C.nets.initial_beliefs)

    mcts_nodes = {n: deepcopy(initial_node) for n in rl_pids}

    def get_and_update_mcts_tree(pid: int, action: int) -> Node:
        node = mcts_nodes[pid].get_action_subtree_and_prune_above(action)
        ensure_visit_count(node, C.mcts.iterations_value_estimate)
        mcts_nodes[pid] = node
        return node

    state = C.game.instance.new_initial_state()
    trajectories = {n: [] for n in rl_pids}

    for pid in rl_pids:
        players[pid].reset_new_game()

    actions = []

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
        if pid in rl_pids:
            obs = state.observation
            action, old_beliefs, root_node = players[pid].request_action(obs)
            actions.append(action)
            target_policy = C.mcts.get_node_target_policy(root_node)
            mcts_nodes[pid] = root_node

            def add_mcts_histogram(label:str, f: Callable[Node, int]):
                fig=plt.figure()
                plt.bar(range(C.game.instance.max_num_actions), list(map(f, root_node.children)))
                sw.add_figure(f"mcts step {step}/{label}", fig,  n)

            if n % 20 == 0:
                with contextlib.suppress(TypeError):
                    add_mcts_histogram("visit count", lambda n: n.visit_count)
                    add_mcts_histogram("value", lambda n: n.value)
                    add_mcts_histogram("reward", lambda n: n.reward)
                    add_mcts_histogram("prior", lambda n: n.prior)
                    #add_mcts_histogram("ucb score", C.mcts.get_node_selection_score)

                    sw.add_histogram(f"mcts step {step}/values", np.array([c.value for c in root_node.children]),n)
                    sw.add_histogram(f"mcts step {step}/visit counts", np.array([c.visit_count for c in root_node.children]),n)
                    sw.add_histogram(f"mcts step {step}/rewards", np.array([c.reward for c in root_node.children]),n)
                    sw.add_histogram(f"mcts step {step}/value preds", np.array([c.value_pred for c in root_node.children]),n)
        else:
            action = players[pid].request_action(state, C.game.instance)

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
                node = get_and_update_mcts_tree(tid, action)
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

    for traj in trajectories.values():
        replay_buffer.add_trajectory(traj, game_terminated=state.is_terminal)

    log.info(f"Finished game ({step + 1} steps)")
    sw.add_scalar("game/length", step + 1, n)
    sw.add_histogram("game/actions", np.array(actions), n)
