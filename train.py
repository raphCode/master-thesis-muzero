import torch
import torch.nn.functional as F
from attrs import Factory, define

from config import config as C
from trajectory import TrajectoryState

tensor_factory = Factory(lambda: torch.tensor(0.0))


@define
class Losses:
    latent: torch.Tensor = tensor_factory
    value: torch.Tensor = tensor_factory
    reward: torch.Tensor = tensor_factory
    policy: torch.Tensor = tensor_factory
    beliefs: torch.Tensor = tensor_factory
    player_type: torch.Tensor = tensor_factory


def process_trajectory(traj: list[TrajectoryState], losses: Losses):
    # TODO: move tensors to GPU

    first = traj[0]
    if first.observation is None:
        beliefs = first.dyn_beliefs
        latent_rep = first.latent_rep
    else:
        latent_rep, beliefs = C.nets.representation(first.observation, first.old_beliefs)

    for ts in traj:
        latent_rep, beliefs, reward = C.nets.dynamics(latent_rep, beliefs, ts.action)
        losses.reward += F.mse_loss(reward, ts.reward)

        if ts.observation is not None:
            new_latent_rep, new_beliefs = C.nets.representation(ts.observation, beliefs)
            losses.latent += F.mse_loss(latent_rep, new_latent_rep)
            losses.beliefs += F.mse_loss(beliefs, new_beliefs)
            latent_rep = new_latent_rep
            beliefs = new_beliefs

        value, policy, player_type = C.nets.prediction(latent_rep, beliefs)
        losses.value += F.mse_loss(value, ts.value)
        losses.policy += F.mse_loss(policy, ts.target_policy)
        # TODO: correct for class imbalance?
        losses.player_type += F.cross_entropy(player_type, ts.player_type)
