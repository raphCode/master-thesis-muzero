import operator
import functools
import itertools
from typing import Callable, Optional, cast

import attrs
import torch
import torch.nn.functional as F
from attrs import Factory, define
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter  # type: ignore [attr-defined]

from config import C
from trajectory import TrainingData
from config.schema import LossWeights
from networks.bases import Networks

zero_tensor = Factory(functools.partial(torch.zeros, 1))


@define
class Losses:
    latent: Tensor = zero_tensor
    value: Tensor = zero_tensor
    reward: Tensor = zero_tensor
    policy: Tensor = zero_tensor
    player: Tensor = zero_tensor

    def weighted_sum(self, weights: LossWeights) -> Tensor:
        """
        Multiplies each loss component with the same-named weight and returns the sum.
        """
        field_names = [f.name for f in attrs.fields(type(self))]
        get_fields = operator.attrgetter(*field_names)  # access fields in same order
        return cast(
            Tensor, sum(l * w for l, w in zip(get_fields(self), get_fields(weights)))
        )


@define
class LossCounts:
    """
    Counts how many losses were calculated in the batch.
    For calculating averages independent of batch size and trajectory length.
    """

    data: int = 0
    latent: int = 0

    def __rtruediv__(self, losses: Losses) -> Losses:
        """
        Divides loss components by the same-named count, or as a fallback the data count.
        Returns a new Losses instance with new tensors.
        """
        values = attrs.asdict(losses)
        for name, loss in values.items():
            divisor = getattr(self, name, self.data)
            values[name] = loss / max(1, divisor)
        return Losses(**values)


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

    def process_batch(self, batch: list[TrainingData], sw: SummaryWriter, n: int) -> None:
        # TODO: move tensors to GPU

        def ml(
            loss_fn: Callable[[Tensor, Tensor], Tensor],
            prediction: Tensor,
            target: Tensor,
            mask: Optional[Tensor] = None,
        ) -> Tensor:
            """
            Masked loss, with the mask defaulting to the current step's is_data.
            loss_fn is expected to not perform any data reduction:
            The results are summed over the batch dimension to calculate a final average
            after summing all unroll steps.
            The average is calculated over the remaining dimensions.
            """
            if mask is None:
                mask = step.is_data
            loss = loss_fn(prediction[mask], target[mask])
            return loss.view(loss.shape[0], -1).mean(dim=1).sum()

        l_pdist = functools.partial(F.pairwise_distance, p=C.training.latent_dist_pnorm)
        l_cross = functools.partial(F.cross_entropy, reduction="none")
        l_mse = functools.partial(F.mse_loss, reduction="none")

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
                if step is first:
                    latent[step.is_observation] = obs_latent[step.is_observation]
                else:
                    losses.latent += ml(
                        l_pdist,
                        latent.flatten(start_dim=1),
                        obs_latent.flatten(start_dim=1),
                        mask=step.is_observation,
                    )
                    counts.latent += cast(int, step.is_observation.count_nonzero().item())

            counts.data += cast(int, step.is_data.count_nonzero().item())

            value, policy_logits, curr_player_logits = self.nets.prediction(
                latent, belief, logits=True
            )
            losses.value += ml(l_mse, value, step.value_target)
            losses.policy += ml(l_cross, policy_logits, step.target_policy)
            losses.player += ml(l_cross, curr_player_logits, step.current_player)

            latent, belief, reward = self.nets.dynamics(
                latent, belief, step.action_onehot
            )
            losses.reward += ml(l_mse, reward, step.reward)

        losses /= counts

        for k, loss in attrs.asdict(losses).items():
            sw.add_scalar(f"loss/{k}", loss, n)  # type: ignore [no-untyped-call]

        total_loss = losses.weighted_sum(C.training.loss_weights)
        self.optimizer.zero_grad()
        total_loss.backward()  # type: ignore [no-untyped-call]
        self.optimizer.step()
        sw.add_scalar("loss/total", total_loss, n)  # type: ignore [no-untyped-call]
