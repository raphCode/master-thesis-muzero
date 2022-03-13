from config import config as C

state = C.game.new_initial_state()
for _ in range(C.param.max_steps_per_episode):
    if state.is_terminal:
        break
    # TODO:
    state.current_player
    # request action from it (mcts treesearch)
    C.nets.representation.forward(state)
    



