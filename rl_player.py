from typing import NamedTuple

import torch

from mcts import run_mcts


class RLPResult(NamedTuple):
    action: int
    old_beliefs: torch.Tensor
    target_policy: torch.Tensor


class RLPlayer:
    """
    Player which does MCTS with neural networks for move selection.
    Don't share a single RLPlayer instance for multiple players because of the private
    beliefs state.
    """

    beliefs: torch.Tensor

    def __init__(self):
        self.reset_new_game()

    def request_action(self, obs: torch.Tensor) -> RLPResult:
        """
        Requests an action from RL agent for the current observation.
        Returns an action to choose as well as extra data for recording a game trajectory.
        """
        old_beliefs = self.beliefs.to(device="cpu", copy=True)
        latent_rep, self.beliefs = C.nets.representation(obs, self.beliefs)
        root_node = run_mcts(latent_rep, new_beliefs)
        action = C.func.mcts_root2action(root_node)
        target_policy = C.func.mcts_root2target_policy(root_node)
        return RLPResult(action, old_beliefs, target_policy)

    def reset_new_game(self):
        self.beliefs = torch.zeros(tuple(C.param.belief_size))
