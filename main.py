import numpy as np

from game import CHANCE_PLAYER_ID
from config import config as C

rng = np.random.default_rng()

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
