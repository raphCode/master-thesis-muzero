import numpy as np
import torch

from game import CHANCE_PLAYER_ID
from config import config as C
from rl_player import RLPlayer
from trajectory import PlayerType, ReplayBuffer, TrajectoryState

rng = np.random.default_rng()

players = ...
rl_pids = {n for n, p in players.items() if isinstance(p, RLPlayer)}
replay_buffer = ReplayBuffer(C.param.replay_buffer_size)

state = C.game.new_initial_state()
trajectories = {n: [] for n in rl_pids}

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
            traj.append(
                TrajectoryState(
                    None,
                    None,
                    PlayerType.Chance,
                    action,
                    chance_outcomes,
                    C.func.calculate_reward(state.rewards, tid),
                )
            )
        continue

    pid = state.current_player
    player = players[pid]
    if pid in rl_pids:
        obs = state.observation
        action, old_beliefs, root_node = player.request_action(obs)
        target_policy = C.func.mcts_root2target_policy(root_node)
    else:
        action = player.request_action(state, C.game)

    # estimate opponent behavior by averaging over their single moves:
    move_onehot = F.one_hot(torch.tensor(action), C.game.max_num_actions)
    state.apply_action(action)

    for tid, traj in trajectories.items():
        if tid == pid:
            ts = TrajectoryState(
                obs,
                old_beliefs,
                PlayerType.Self,
                action,
                target_policy,
                C.func.calculate_reward(state.rewards, pid),
            )
        else:
            ts = TrajectoryState(
                None,
                None,
                PlayerType.Teammate
                if C.func.is_teammate(pid, tid)
                else PlayerType.Opponent,
                action,
                move_onehot,
                C.func.calculate_reward(state.rewards, tid),
            )
        traj.append(ts)

for traj in trajectories:
    replay_buffer.add_trajectory(traj)
