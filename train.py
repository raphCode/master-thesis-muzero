import attrs
import torch
import torch.nn.functional as F
from attrs import define
from torch.utils.tensorboard import SummaryWriter

from config import C
from trajectory import TrainingData
from networks.bases import Networks


@define
class Losses:
    latent: torch.Tensor = 0
    value: torch.Tensor = 0
    reward: torch.Tensor = 0
    policy: torch.Tensor = 0
    player_type: torch.Tensor = 0


@define
class LossCounts:
    """
    Counts how many losses were calculated in the batch.
    For calculating averages independent of batch size and trajectory length.
    """

    data: int = 0
    latent: int = 0


class Trainer:
    def __init__(self, nets: Networks):
        self.nets = nets
        self.optimizer = C.train.optimizer_factory(self.nets)

    def process_batch(self, batch: list[TrainingData], sw: SummaryWriter, n: int):
        # TODO: move tensors to GPU
        losses = Losses()
        counts = LossCounts()

        first = batch[0]
        obs_latent_rep, obs_beliefs = self.nets.representation(
            *first.observation, first.beliefs
        )
        latent_rep = obs_latent_rep.where(
            first.is_observation.unsqueeze(-1), first.latent_rep
        )
        beliefs = obs_beliefs.where(first.is_observation.unsqueeze(-1), first.beliefs)

        for step in batch:
            if not step.is_data.any():
                break

            if step is not first and step.is_observation.any():
                obs_latent_rep, obs_beliefs = self.nets.representation(
                    *step.observation, beliefs
                )
                losses.latent += (
                    F.mse_loss(latent_rep, obs_latent_rep, reduction="none")
                    .mean(dim=1)
                    .masked_select(step.is_observation)
                    .sum()
                )
                counts.latent += step.is_observation.count_nonzero()

            counts.data += step.is_data.count_nonzero()

            value, policy_logits, player_type_logits = self.nets.prediction(
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

            latent_rep, beliefs, reward = self.nets.dynamics(
                latent_rep, beliefs, step.action_onehot
            )
            losses.reward += F.mse_loss(reward, step.reward, reduction="none")[
                step.is_data
            ].sum()

        if counts.latent > 0:
            losses.latent /= counts.latent
        if counts.data > 0:
            losses.value /= counts.data
            losses.policy /= counts.data
            losses.reward /= counts.data
            losses.player_type /= counts.data

        for k, loss in attrs.asdict(losses).items():
            sw.add_scalar(f"loss/{k}", loss, n)

        loss = (
            C.training.loss_weights.latent * losses.latent
            + C.training.loss_weights.value * losses.value
            + C.training.loss_weights.reward * losses.reward
            + C.training.loss_weights.policy * losses.policy
            + C.training.loss_weights.player_type * losses.player_type
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        sw.add_scalar("loss/total", loss, n)
