from config import config as C

state = C.game.new_initial_state()
for _ in range(C.param.max_steps_per_episode):
    # Unsure about how to deal with non-terminal rewards or when exactly they occur
    assert all(r == 0 for r in state.rewards) or state.is_terminal

    if state.is_terminal:
        break
    # TODO:
    state.current_player
    # request action from it (mcts treesearch)
    C.nets.representation.forward(state)
