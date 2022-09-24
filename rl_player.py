from typing import NamedTuple

import torch

from mcts import run_mcts
from config import C
from globals import G


class RLPResult(NamedTuple):
    action: int
    old_beliefs: torch.Tensor
    mcts_value: float


class RLPlayer:
    """
    Player which does MCTS with neural networks for move selection.
    Don't share a single RLPlayer instance for multiple players because of the private
    beliefs state.
    """

    beliefs: torch.Tensor
    move_number: int

    def request_action(self, observation: tuple[torch.Tensor]) -> RLPResult:
        """
        Requests an action from RL agent for the current observation.
        Returns an action to choose as well as extra data for recording a game trajectory.
        """
        old_beliefs = self.beliefs.to(device="cpu", copy=True)
        latent_rep, self.beliefs = G.nets.representation.si(*observation, self.beliefs)
        # TODO: instead of re-running, try to reuse previous tree search from selfplay here
        root_node = run_mcts(latent_rep, self.beliefs)
        action = C.mcts.get_node_action(root_node, self.move_number)
        self.move_number += 1
        return RLPResult(action, old_beliefs, root_node)

    def reset_new_game(self):
        self.beliefs = G.nets.initial_beliefs.detach().clone()
        self.move_number = 0
