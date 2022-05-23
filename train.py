import torch.nn.functional as F

from trajectory import TrajectoryState


def process_trajectory(traj: List[TrajectoryState]):
    # TODO: move tensors to GPU
    latent_loss = ...

    beliefs = traj[0].beliefs
    latent_rep = None
    for ts in traj:
        if ts.observation is not None:
            new_latent_rep, beliefs = C.nets.representation(ts.observation, beliefs)
            if latent_rep is not None and C.param.efficient_zero_optimisation:
                latent_loss += F.mse_loss(latent_rep, new_latent_rep)
            latent_rep = new_latent_rep
