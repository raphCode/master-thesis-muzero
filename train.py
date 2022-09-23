import attrs
import torch
import torch.nn.functional as F
from attrs import define
from torch.utils.tensorboard import SummaryWriter

from config import C
from trajectory import TrainingData


@define
class Losses:
    latent: torch.Tensor = 0
    value: torch.Tensor = 0
    reward: torch.Tensor = 0
    policy: torch.Tensor = 0
    player_type: torch.Tensor = 0


@define
class LossDataCounts:
    fit: int = 0
    latent: int = 0


def process_batch(batch: list[TrainingData], sw: SummaryWriter, n: int):
    # TODO: move tensors to GPU
    losses = Losses()
    counts = LossDataCounts()

    first = batch[0]
    obs_latent_rep, obs_beliefs = C.nets.representation(*first.observation, first.beliefs)
    latent_rep = obs_latent_rep.where(
        first.is_observation.unsqueeze(-1), first.latent_rep
    )
    beliefs = obs_beliefs.where(first.is_observation.unsqueeze(-1), first.beliefs)

    for step in batch:
        if not step.is_data.any():
            break

        if step is not first and step.is_observation.any():
            obs_latent_rep, obs_beliefs = C.nets.representation(
                *step.observation, beliefs
            )
            losses.latent += (
                F.mse_loss(latent_rep, obs_latent_rep, reduction="none")
                .mean(dim=1)
                .masked_select(step.is_observation)
                .sum()
            )
            counts.latent += step.is_observation.count_nonzero()

        counts.fit += step.is_data.count_nonzero()

        value, policy_logits, player_type_logits = C.nets.prediction(
            latent_rep, beliefs, logits=True
        )
        losses.value += F.mse_loss(value, step.value_target, reduction="none")[
            step.is_data
        ].sum()
        losses.policy += F.cross_entropy(
            policy_logits, step.target_policy, reduction="none"
        )[step.is_data].sum()
        losses.player_type += F.cross_entropy(
            player_type_logits, step.player_type, reduction="none"
        )[step.is_data].sum()

        latent_rep, beliefs, reward = C.nets.dynamics(
            latent_rep, beliefs, step.action_onehot
        )
        losses.reward += F.mse_loss(reward, step.reward, reduction="none")[
            step.is_data
        ].sum()

    if counts.latent > 0:
        losses.latent /= counts.latent
    if counts.fit > 0:
        losses.value /= counts.fit
        losses.policy /= counts.fit
        losses.reward /= counts.fit
        losses.player_type /= counts.fit

    for k, loss in attrs.asdict(losses).items():
        sw.add_scalar(f"loss/{k}", loss, n)

    loss = (
        C.train.loss_weights.latent * losses.latent
        + C.train.loss_weights.value * losses.value
        + C.train.loss_weights.reward * losses.reward
        + C.train.loss_weights.policy * losses.policy
        + C.train.loss_weights.player_type * losses.player_type
    )
    C.train.optimizer.zero_grad()
    loss.backward()
    C.train.optimizer.step()

    sw.add_scalar("loss/total", loss, n)
