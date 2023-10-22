common_args := "players=single mcts=test "
debug_cmd := "pudb main.py hydra.run.dir=/tmp/hydra_dbg " + common_args
train_cmd := "python main.py " + common_args

[private]
clear-screen:
    clear

train-catch *params: clear-screen
    {{train_cmd}} training=test networks=test game=os_catch {{params}}

debug-catch *params: && clear-screen
    {{debug_cmd}} training=test networks=test game=os_catch {{params}}
