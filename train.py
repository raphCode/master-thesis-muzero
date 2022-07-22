import torch
import torch.nn.functional as F
from attrs import define, Factory

from config import config as C
from trajectory import PlayerType, TrajectoryState

tensor_factory = Factory(lambda:torch.tensor(0.0))

@define
class Loss:
    latent: torch.Tensor = tensor_factory
    value: torch.Tensor = tensor_factory
    reward: torch.Tensor = tensor_factory
    policy: torch.Tensor = tensor_factory
    beliefs: torch.Tensor = tensor_factory
    player_type: torch.Tensor = tensor_factory


def process_trajectory(traj: list[TrajectoryState], loss: Loss):
    # TODO: move tensors to GPU

    first = traj[0]
    if first.observation is None:
        beliefs = first.dyn_beliefs
        latent_rep = first.latent_rep
    else:
        latent_rep, beliefs = C.nets.representation.si(
            first.observation, first.old_beliefs
        )

    for ts in traj:
        latent_rep, beliefs, reward = C.nets.dynamics.si(
            latent_rep,
            beliefs,
            F.one_hot(torch.tensor(ts.action), C.game.instance.max_num_actions),
        )
        loss.reward += F.mse_loss(reward, torch.tensor(ts.reward, dtype=torch.float).view(1))

        if ts.observation is not None:
            new_latent_rep, new_beliefs = C.nets.representation.si(
                ts.observation, beliefs
            )
            loss.latent += F.mse_loss(latent_rep, new_latent_rep)
            loss.beliefs += F.mse_loss(beliefs, new_beliefs)
            latent_rep = new_latent_rep
            beliefs = new_beliefs

        value, policy, player_type = C.nets.prediction.si(latent_rep, beliefs, logits=True)
        loss.value += F.mse_loss(value, torch.tensor(ts.value, dtype=torch.float).view(1))
        loss.policy += F.cross_entropy(policy.unsqueeze(0), torch.tensor(ts.target_policy, dtype=torch.float).unsqueeze(0)
        )
        # TODO: correct for class imbalance?
        loss.player_type += F.cross_entropy(player_type, torch.tensor(ts.player_type,dtype=int)
        )


def process_batch(batch: list[list[TrajectoryState]]):
    torch.autograd.set_detect_anomaly(True)
    losses = Loss()
    for traj in batch:
        # TODO: batch network inference
        process_trajectory(traj, losses)

    import attrs

    rate = C.train.loss_weights.automatic_adjustment_rate
    def adj(x, l):
        return x * (1-rate) + 1/(l+1e-3)* rate
    
    for k, loss in attrs.asdict(losses).items():
        lw=getattr(C.train.loss_weights, k)
        setattr(C.train.loss_weights, k, adj(lw, loss.item()))
        print(f"{k}: {lw:.4f}, loss {loss:.4f}")

    loss = (
        C.train.loss_weights.latent * losses.latent
        + C.train.loss_weights.value * losses.value
        + C.train.loss_weights.reward * losses.reward
        + C.train.loss_weights.policy * losses.policy
        + C.train.loss_weights.beliefs * losses.beliefs
        + C.train.loss_weights.player_type * losses.player_type
    )
    loss.backward()
    C.train.optimizer.step()

    return loss

    for k, x in attrs.asdict(losses).items():
        print(k, x)

    return loss
