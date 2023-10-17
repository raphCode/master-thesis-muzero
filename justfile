debug-catch:
  pudb main.py mcts=default training=catch networks=catch game=os_catch hydra.run.dir=/tmp/hydr_dbg; clear

train-catch:
  clear; python main.py mcts=default training=catch networks=catch game=os_catch

debug-carchess:
  pudb main.py mcts=default training=carchess networks=carchess game=carchess training.batch_size=64 training.replay_buffer_size=5000 hydra.run.dir=/tmp/hydr_dbg; clear

train-carchess:
  clear; python main.py mcts=default training=carchess networks=carchess game=carchess

debug-2048:
  pudb main.py mcts=default training=\"2048\" networks=\"2048\" game=os_2048 hydra.run.dir=/tmp/hydr_dbg; clear

train-2048:
  clear; python main.py mcts=default training=\"2048\" networks=\"2048\" game=os_2048
