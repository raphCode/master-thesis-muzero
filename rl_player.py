from typing import NamedTuple

import torch


class RLPResult(NamedTuple):
    action: int
    old_beliefs: torch.Tensor
    mcts_value: float
