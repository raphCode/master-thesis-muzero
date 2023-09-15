debug-2048:
  pudb main.py mcts=default training=\"2048\" networks=\"2048\" game=os_2048 players=single; clear

train-2048:
  clear; python main.py mcts=default training=\"2048\" networks=\"2048\" game=os_2048 players=single
