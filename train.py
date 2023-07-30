from __future__ import annotations

import logging
import operator
import functools
from typing import TYPE_CHECKING, Callable, Optional, cast

import attrs
import torch
from attrs import Factory, define
from torch import Tensor, nn

from config import C

if TYPE_CHECKING:
    from networks import Networks
    from trajectory import TrainingData
    from config.schema import LossWeights
    from tensorboard_wrapper import TBStepLogger


zero_tensor = Factory(functools.partial(torch.zeros, 1))


log = logging.getLogger(__name__)


@define
class Losses:
    """
    The different loss components, one for each training objective.
    """

    value: Tensor = zero_tensor
    latent: Tensor = zero_tensor
    reward: Tensor = zero_tensor
    policy: Tensor = zero_tensor
    turn: Tensor = zero_tensor

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
        lrs = C.training.learning_rates
        self.optimizer = C.training.optimizer(
            [
                dict(
                    params=nets.representation.parameters(),
                    lr=lrs.base * lrs.representation,
                ),
                dict(
                    params=nets.prediction.parameters(),
                    lr=lrs.base * lrs.prediction,
                ),
                dict(
                    params=nets.dynamics.parameters(),
                    lr=lrs.base * lrs.dynamics,
                ),
                dict(
                    params=[nets.initial_latent],
                    lr=lrs.base * lrs.initial_tensors,
                ),
            ]
        )

    def process_batch(self, batch: list[TrainingData], tbs: TBStepLogger) -> None:
        # TODO: move tensors to GPU

        def ml(
            criterion: Callable[[Tensor, Tensor], Tensor],
            prediction: Tensor,
            target: Tensor,
            mask: Optional[Tensor] = None,
        ) -> Tensor:
            """
            Masked loss, with the mask defaulting to the current step's is_data.
            criterion is expected to not perform any data reduction:
            The results are summed over the batch dimension to calculate a final average
            after summing all unroll steps.
            The average is calculated over the remaining loss result dimensions.
            """
            if mask is None:
                mask = step.is_data
            loss = criterion(prediction[mask], target[mask])
            return loss.view(loss.shape[0], -1).mean(dim=1).sum()

        pdist = nn.PairwiseDistance(p=C.training.latent_dist_pnorm)
        cross = nn.CrossEntropyLoss(reduction="none")
        mse = nn.MSELoss(reduction="none")

        losses = Losses()
        counts = LossCounts()

        first = batch[0]
        latent = first.latent
        latent[first.is_initial] = self.nets.initial_latent

        for step in batch:
            if step.is_observation.any():
                obs_latent = self.nets.representation(*step.observations)
                if step is first:
                    latent[step.is_observation] = obs_latent[step.is_observation]
                else:
                    losses.latent += ml(
                        pdist,
                        latent.flatten(start_dim=1),
                        obs_latent.flatten(start_dim=1),
                        mask=step.is_observation,
                    )
                    counts.latent += cast(int, step.is_observation.count_nonzero().item())

            counts.data += cast(int, step.is_data.count_nonzero().item())

            value, policy_logits = self.nets.prediction.raw_forward(
                latent,
            )
            value_target = self.nets.prediction.value_scale.get_target(step.value_target)
            losses.value += ml(mse, value, value_target)
            losses.policy += ml(cross, policy_logits, step.target_policy)

            latent, reward, turn_status_logits = self.nets.dynamics.raw_forward(
                latent,
                step.action_onehot,
            )
            reward_target = self.nets.dynamics.reward_scale.get_target(step.reward)
            losses.reward += ml(mse, reward, reward_target)
            losses.turn += ml(cross, turn_status_logits, step.turn_status)

        losses /= counts

        for k, loss in attrs.asdict(losses).items():
            tbs.add_scalar(f"loss/{k}", loss)

        total_loss = losses.weighted_sum(C.training.loss_weights)
        self.optimizer.zero_grad()
        total_loss.backward()  # type: ignore [no-untyped-call]
        self.optimizer.step()

        log.info(
            "Finished batch update, traj length {}, loss {:.3f}".format(
                len(batch),
                total_loss.item(),
            )
        )
