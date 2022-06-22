from typing import NamedTuple

import torch
import torch.nn.functional as F

from config import config as C
from trajectory import TrajectoryState


class Loss(NamedTuple):
    latent: torch.Tensor
    value: torch.Tensor
    reward: torch.Tensor
    policy: torch.Tensor
    beliefs: torch.Tensor
    player_type: torch.Tensor


def process_trajectory(traj: list[TrajectoryState], loss: Loss):
    # TODO: move tensors to GPU

    first = traj[0]
    if first.observation is None:
        beliefs = first.dyn_beliefs
        latent_rep = first.latent_rep
    else:
        latent_rep, beliefs = C.nets.representation(first.observation, first.old_beliefs)

    for ts in traj:
        latent_rep, beliefs, reward = C.nets.dynamics(latent_rep, beliefs, ts.action)
        loss.reward += F.mse_loss(reward, ts.reward)

        if ts.observation is not None:
            new_latent_rep, new_beliefs = C.nets.representation(ts.observation, beliefs)
            loss.latent += F.mse_loss(latent_rep, new_latent_rep)
            loss.beliefs += F.mse_loss(beliefs, new_beliefs)
            latent_rep = new_latent_rep
            beliefs = new_beliefs

        value, policy, player_type = C.nets.prediction(latent_rep, beliefs)
        loss.value += F.mse_loss(value, ts.value)
        loss.policy += F.mse_loss(policy, ts.target_policy)
        # TODO: correct for class imbalance?
        loss.player_type += F.cross_entropy(player_type, ts.player_type)
