common_args := "mcts=default "
debug_cmd := "HYDRA_FULL_ERROR=1 pudb main.py hydra.run.dir=/tmp/hydra_dbg " + common_args
train_cmd := "python main.py " + common_args
bisect_run_dir := "'hydra.run.dir=bisect/${now:%Y-%m-%d}/${now:%H-%M-%S}-'`git rev-parse --short HEAD`"

[private]
clear-screen:
    clear

train-catch *params: clear-screen
    {{train_cmd}} training=catch networks=catch_cnn game=os_catch {{params}}

debug-catch *params: && clear-screen
    {{debug_cmd}} training=catch networks=catch game=os_catch {{params}}

bisect-catch: (train-catch bisect_run_dir)

train-2048 *params: clear-screen
    {{train_cmd}} training=\"2048\" networks=\"2048\" game=os_2048 {{params}}

train-carchess *params: clear-screen
    {{train_cmd}} training=carchess networks=carchess game=carchess {{params}}

debug-carchess *params: clear-screen
    {{debug_cmd}} training=carchess networks=carchess game=carchess {{params}}
