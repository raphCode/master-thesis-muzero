import numpy as np
import torch

from game import CHANCE_PLAYER_ID
from config import config as C
from rl_player import RLPlayer

rng = np.random.default_rng()

players = ...

state = C.game.new_initial_state()
for _ in range(C.param.max_steps_per_episode):
    # Unsure about how to deal with non-terminal rewards or when exactly they occur
    assert all(r == 0 for r in state.rewards) or state.is_terminal

    if state.is_terminal:
        break

    pid = state.current_player
    if pid == CHANCE_PLAYER_ID:
        chance_outcomes = state.chance_outcomes
        action = rng.choice(C.game.max_num_actions, p=chance_outcomes)
        state.apply_action(action)
        continue

    player = players[pid]
    if isinstance(player, RLPlayer):
        obs = state.observation
        action, old_beliefs, mcts_policy = player.request_action(obs)
    else:
        action = player.request_action(state, C.game)

    state.apply_action(action)
