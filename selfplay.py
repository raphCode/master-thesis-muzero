import logging
from typing import Callable
from contextlib import suppress

import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from mcts import Node, ensure_visit_count
from config import config as C
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


def run_episode(replay_buffer: ReplayBuffer, sw: SummaryWriter, n: int):
    C.nets.dynamics.eval()
    C.nets.representation.eval()

    players = C.player.instances
    rl_pids = {n for n, p in enumerate(players) if isinstance(p, RLPlayer)}

    mcts_nodes = {
        n: Node.from_latents(C.nets.initial_latent_rep, C.nets.initial_beliefs)
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

    actions = []
    latents = []
    reward0 = 0

    for step in range(C.train.max_steps_per_episode):
        # Unsure about how to deal with non-terminal rewards or when exactly they occur
        assert state.is_chance or state.is_terminal or all(r == 0 for r in state.rewards)

        reward0 += C.game.calculate_reward(state.rewards, 0)

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
            actions.append(action)
            target_policy = C.mcts.get_node_target_policy(root_node)
            mcts_nodes[pid] = root_node

            latents.append(root_node.latent_rep)

            def debug_mcts_node(node: Node):
                actions = []
                nn = node
                while nn != root_node:
                    actions.append(str(nn.action))
                    nn = nn.parent
                path = "_".join(reversed(actions)) or "root"

                def add_mcts_histogram(label: str, f: Callable[Node, int]):
                    values = []
                    actions = []
                    unexpanded = []
                    for child in node.children:
                        if child.is_expanded:
                            values.append(f(child))
                            actions.append(child.action)
                        else:
                            unexpanded.append(child.action)
                    fig = plt.figure()
                    plt.bar(
                        actions,
                        values,
                    )
                    plt.scatter(
                        unexpanded,
                        [0] * len(unexpanded),
                        s=500,
                        color="red",
                        marker="x",
                        linewidth=5,
                    )
                    sw.add_figure(f"mcts plot {path}/{label}", fig, n)
                    if len(values) > 0:
                        sw.add_histogram(f"mcts {path}/{label}", np.array(values), n)

                add_mcts_histogram("visit count", lambda n: n.visit_count)
                add_mcts_histogram("prior", lambda n: n.prior)
                add_mcts_histogram("value", lambda n: n.value)
                add_mcts_histogram("combined value", lambda n: n.value + n.reward)
                add_mcts_histogram(
                    "target policy",
                    lambda n: C.mcts.get_node_target_policy(n.parent)[n.action],
                )
                add_mcts_histogram("reward", lambda n: n.reward)
                # add_mcts_histogram("node selection score", C.mcts.get_node_selection_score)

            if step == 7 and n % 200 == 0 and C.tb:
                fig = plt.figure()
                plt.imshow(obs[0])
                sw.add_figure(f"game/observation 0", fig, n)

                import pickle

                with open(f"tree_{n:06d}.pk", "wb") as f:
                    with suppress(RecursionError):
                        pickle.dump(root_node, f)

                debug_mcts_node(root_node)
                for aa, child in enumerate(root_node.children):
                    debug_mcts_node(child)
                    for nn, cc in enumerate(child.children):
                        debug_mcts_node(cc)

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
    sw.add_scalar("game/length", step + 1, n)
    sw.add_scalar("game/cumultative reward player 0", reward0, n)
    sw.add_histogram("game/actions", np.array(actions), n)

    with suppress(ValueError):
        sw.add_histogram(f"latent variance", torch.stack(latents).var(dim=0), n)
        sw.add_scalar(f"latent variance/max", torch.stack(latents).var(dim=0).max(), n)
        sw.add_scalar(f"latent variance/mean", torch.stack(latents).var(dim=0).mean(), n)
    for i, x in enumerate(torch.stack(latents).T[:10]):
        with suppress(ValueError):
            sw.add_histogram(f"latent space throughout game/dim {i}", x, n)
            sw.add_scalar(f"latent space throughout game variance/dim {i}", x.var(), n)
            sw.add_histogram(f"initials/latent rep", C.nets.initial_latent_rep, n)
