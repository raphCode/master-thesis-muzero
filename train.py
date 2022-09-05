import torch
import torch.nn.functional as F
from attrs import Factory, define

from config import config as C
from trajectory import TrajectoryState


@define
class Losses:
    latent: torch.Tensor = 0
    value: torch.Tensor = 0
    reward: torch.Tensor = 0
    policy: torch.Tensor = 0
    player_type: torch.Tensor = 0


def process_trajectory(traj: list[TrajectoryState], losses: Losses):
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
        losses.reward += F.l1_loss(
            reward, torch.tensor(ts.reward, dtype=torch.float).view(1)
        )

        if ts.observation is not None:
            new_latent_rep, new_beliefs = C.nets.representation.si(
                ts.observation, beliefs
            )
            losses.latent += F.cosine_embedding_loss(
                latent_rep, new_latent_rep, torch.tensor(1)
            )
            # TODO: losses here are not properly averaged: this branch is not taken for all items in the batch

        value, policy, player_type = C.nets.prediction.si(
            latent_rep, beliefs, logits=True
        )
        losses.value += F.l1_loss(
            value, torch.tensor(ts.value, dtype=torch.float).view(1)
        )
        losses.policy += F.cross_entropy(
            policy.unsqueeze(0),
            torch.tensor(ts.target_policy, dtype=torch.float).unsqueeze(0),
        )
        # TODO: correct for class imbalance?
        losses.player_type += F.cross_entropy(
            player_type, torch.tensor(ts.player_type, dtype=int)
        )


def process_batch(batch: list[list[TrajectoryState]], sw: SummaryWriter, n: int):
    losses = Losses()
    for traj in batch:
        # TODO: batch network inference
        process_trajectory(traj, losses)

    for k, loss in attrs.asdict(losses).items():
        sw.add_scalar(f"loss/{k}", loss, n)

    loss = (
        C.train.loss_weights.latent * losses.latent
        + C.train.loss_weights.value * losses.value
        + C.train.loss_weights.reward * losses.reward
        + C.train.loss_weights.policy * losses.policy
        + C.train.loss_weights.player_type * losses.player_type
    ) / bsize
    C.train.optimizer.zero_grad()
    loss.backward()
    C.train.optimizer.step()

    sw.add_scalar("loss/total", loss, n)
