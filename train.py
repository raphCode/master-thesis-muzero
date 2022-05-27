from typing import NamedTuple

import torch.nn.functional as F

from config import config as C
from trajectory import TrajectoryState


class Loss(NamedTuple):
    latent: torch.Tensor
    value: torch.Tensor
    policy: torch.Tensor
    player_type: torch.Tensor


def process_trajectory(traj: List[TrajectoryState], loss: Loss):
    # TODO: move tensors to GPU

    beliefs = traj[0].beliefs
    latent_rep = None
    for ts in traj:
        if ts.observation is not None:
            new_latent_rep, beliefs = C.nets.representation(ts.observation, beliefs)
            if latent_rep is not None and C.param.efficient_zero_optimisation:
                loss.latent += F.mse_loss(latent_rep, new_latent_rep)
            latent_rep = new_latent_rep
