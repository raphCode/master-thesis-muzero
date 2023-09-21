debug-catch:
  pudb main.py mcts=default training=catch networks=catch game=os_catch players=single; clear

train-catch:
  clear; python main.py mcts=default training=catch networks=catch game=os_catch players=single

debug-carchess:
  pudb main.py mcts=default training=carchess networks=carchess game=carchess players=single; clear

train-carchess:
  clear; python main.py mcts=default training=carchess networks=carchess game=carchess players=single

debug-2048:
  pudb main.py mcts=default training=\"2048\" networks=\"2048\" game=os_2048 players=single; clear

train-2048:
  clear; python main.py mcts=default training=\"2048\" networks=\"2048\" game=os_2048 players=single
