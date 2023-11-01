common_args := "mcts=muzero "
debug_cmd := "HYDRA_FULL_ERROR=1 pudb main.py hydra.run.dir=/tmp/hydra_dbg " + common_args
train_cmd := "python main.py " + common_args
dated_run := "${now:%Y-%m-%d}/${now:%H-%M-%S}"
bisect_run_dir := "'hydra.run.dir=bisect/" + dated_run + "-'`git rev-parse --short HEAD`"

[private]
clear-screen:
    clear

train-catch *params: clear-screen
    {{train_cmd}} training=catch networks=catch game=os_catch {{params}}

debug-catch *params: && clear-screen
    {{debug_cmd}} training=catch networks=catch game=os_catch {{params}}

bisect-catch: (train-catch bisect_run_dir)

train-2048 *params: clear-screen
    {{train_cmd}} training=\"2048\" networks=\"2048\" game=os_2048 {{params}}

eval-ll eval_dir *params: 
    {{train_cmd}} training=catch networks=catch game=os_catch 'hydra.run.dir=outputs/{{eval_dir}}/muzero/{{dated_run}}' {{params}} training.random_unroll_length=true training.loss_weights.latent=0
    {{train_cmd}} training=catch networks=catch game=os_catch 'hydra.run.dir=outputs/{{eval_dir}}/effzero/{{dated_run}}' {{params}} training.random_unroll_length=true training.latent_loss_detach=true
    {{train_cmd}} training=catch networks=catch game=os_catch 'hydra.run.dir=outputs/{{eval_dir}}/raphzero/{{dated_run}}' {{params}} training.random_unroll_length=true training.latent_loss_detach=false

eval-ll-big eval_dir *params: (eval-ll eval_dir "game.instance.columns=10" "training=catch_big" params)

eval-tns eval_dir *params: 
    {{train_cmd}} training=catch networks=catch game=os_catch 'hydra.run.dir=outputs/{{eval_dir}}/muzero/{{dated_run}}' {{params}} training.absorbing_terminal_states=true
    {{train_cmd}} training=catch networks=catch game=os_catch 'hydra.run.dir=outputs/{{eval_dir}}/raphzero/{{dated_run}}' {{params}} training.absorbing_terminal_states=false
