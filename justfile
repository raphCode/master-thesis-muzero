common_args := "players=single mcts=test "
debug_cmd := "pudb main.py hydra.run.dir=/tmp/hydra_dbg " + common_args
train_cmd := "python main.py " + common_args
bisect_run_dir := "'hydra.run.dir=bisect/${now:%Y-%m-%d}/${now:%H-%M-%S}-'`git rev-parse --short HEAD`"

[private]
clear-screen:
    clear

train-catch *params: clear-screen
    {{train_cmd}} training=test networks=test game=os_catch {{params}}

debug-catch *params: && clear-screen
    {{debug_cmd}} training=test networks=test game=os_catch {{params}}

bisect-catch: (train-catch bisect_run_dir)
