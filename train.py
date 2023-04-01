import functools
import itertools

import attrs
import torch
import torch.nn.functional as F
from attrs import Factory, define
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from config import C
from trajectory import TrainingData
from networks.bases import Networks

zero_tensor = Factory(functools.partial(torch.zeros, 1))


@define
class Losses:
    latent: Tensor = zero_tensor
    value: Tensor = zero_tensor
    reward: Tensor = zero_tensor
    policy: Tensor = zero_tensor
    player: Tensor = zero_tensor


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
        self.optimizer = C.training.optimizer(
            itertools.chain(
                nets.representation.parameters(),
                nets.prediction.parameters(),
                nets.dynamics.parameters(),
                [nets.initial_latent],
                [nets.initial_belief] if nets.initial_belief is not None else [],
            )
        )

    def process_batch(self, batch: list[TrainingData], sw: SummaryWriter, n: int):
        # TODO: move tensors to GPU
        losses = Losses()
        counts = LossCounts()

        first = batch[0]
        latent = first.latent
        latent[first.is_initial] = self.nets.initial_latent
        belief = first.belief
        if belief is not None:
            assert self.nets.initial_belief is not None
            belief[first.is_initial] = self.nets.initial_belief

        for step in batch:
            if step.is_observation.any():
                obs_latent = self.nets.representation(*step.observations)
                if step is not first:
                    losses.latent += (
                        F.mse_loss(latent, obs_latent, reduction="none")
                        .mean(dim=1)
                        .masked_select(step.is_observation)
                        .sum()
                    )
                    counts.latent += step.is_observation.count_nonzero()
                    # latent came from the dynamics network,
                    # it might be a tensor view where direct assignment is not possible
                    latent = latent.contiguous()
                latent[step.is_observation] = obs_latent[step.is_observation]

            counts.data += step.is_data.count_nonzero()

            value, policy_logits, curr_player_logits = self.nets.prediction(
                latent, belief, logits=True
            )
            losses.value += F.mse_loss(value, step.value_target, reduction="none")[
                step.is_data
            ].sum()
            losses.policy += F.cross_entropy(
                policy_logits, step.target_policy, reduction="none"
            )[step.is_data].sum()
            losses.player += F.cross_entropy(
                curr_player_logits, step.current_player, reduction="none"
            )[step.is_data].sum()

            latent, belief, reward = self.nets.dynamics(
                latent, belief, step.action_onehot
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
            losses.player /= counts.data

        for k, loss in attrs.asdict(losses).items():
            sw.add_scalar(f"loss/{k}", loss, n)

        loss = (
            C.training.loss_weights.latent * losses.latent
            + C.training.loss_weights.value * losses.value
            + C.training.loss_weights.reward * losses.reward
            + C.training.loss_weights.policy * losses.policy
            + C.training.loss_weights.player * losses.player
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        sw.add_scalar("loss/total", loss, n)
