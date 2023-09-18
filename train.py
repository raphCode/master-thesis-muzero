from __future__ import annotations

import math
import logging
import operator
import functools
from typing import TYPE_CHECKING, Any, Callable, Optional, cast

import attrs
import numpy as np
import torch
from attrs import Factory, define
from toolz import dicttoolz  # type: ignore
from torch import Tensor, nn

from config import C

if TYPE_CHECKING:
    from util import ndarr_f64
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

    def weighted_sum(self, weights: dict[str, float]) -> Tensor:
        """
        Multiplies each loss component with the same-named weight and returns the sum.
        """
        losses = attrs.asdict(self)
        assert losses.keys() == weights.keys()
        return cast(
            Tensor,
            sum(dicttoolz.merge_with(math.prod, losses, weights).values()),
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


class SLAW:
    """
    Implements automatic loss weighting as described in:
    "Scaled Loss Approximate Weighting for Efficient Multi-Task Learning"
    https://arxiv.org/abs/2109.08218
    """

    beta: float
    auto_weight_names: list[str]  # names of loss components / weights, in order
    fixed_weights: dict[str, float]
    a: ndarr_f64
    b: ndarr_f64
    w: ndarr_f64

    def __init__(self, loss_weight_config: LossWeights, mavg_beta: float) -> None:
        def can_parse_float(x: Any) -> bool:
            try:
                float(x)
                return True
            except ValueError:
                return False

        self.auto_weight_names = []
        self.fixed_weights = {}
        for k, v in attrs.asdict(loss_weight_config).items():
            if v == "auto":
                self.auto_weight_names.append(k)
                continue
            assert can_parse_float(v), (
                "Loss weights must either be a float or the string 'auto'!\n"
                f"Could not parse value for weight '{k}': {v}"
            )
            self.fixed_weights[k] = float(v)

        n = len(self.auto_weight_names)
        self.a = np.zeros(n)
        self.b = np.zeros(n)
        self.w = np.ones(n)
        self.beta = mavg_beta

    def step(self, losses: Losses) -> None:
        """
        Update internal loss weights based on the current training step's losses.
        """
        n = len(self.auto_weight_names)
        if n == 0:
            return
        get_fields = operator.attrgetter(*self.auto_weight_names)
        l = np.array([t.item() for t in get_fields(losses)])
        self.a = self.beta * self.a + (1 - self.beta) * l**2
        self.b = self.beta * self.b + (1 - self.beta) * l
        s = np.maximum(1e-5, np.sqrt(self.a - self.b**2))
        self.w = (n / s) / np.sum(1 / s)

    @property
    def weights(self) -> dict[str, float]:
        """
        Return the calculated loss weights.
        """
        return dict(zip(self.auto_weight_names, self.w)) | self.fixed_weights


class Trainer:
    def __init__(self, nets: Networks):
        self.nets = nets
        lrs = C.training.learning_rates
        self.slaw = SLAW(C.training.loss_weights, 0.999)
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
            criterion is expected to not perform any data reduction:
            The results are summed over the batch dimension to calculate a final average
            after summing all unroll steps.
            The average is calculated over the remaining loss result dimensions.
            """
            loss = criterion(prediction, target)
            return loss.view(loss.shape[0], -1).mean(dim=1).sum()

        pdist = nn.PairwiseDistance(p=C.training.latent_dist_pnorm)
        cross = nn.CrossEntropyLoss(reduction="none")

        losses = Losses()
        counts = LossCounts()

        first = batch[0]
        batch_size = len(first.reward)
        latent: Tensor

        for step in batch:
            if step.is_observation.any():
                obs_latent = self.nets.representation(*step.observations)
                if step is first:
                    assert step.is_observation.all()
                    latent = obs_latent
                else:
                    mask = step.is_observation
                    losses.latent += ml(
                        pdist,
                        latent[mask].flatten(start_dim=1),
                        obs_latent[mask].flatten(start_dim=1),
                    )
                    counts.latent += cast(int, step.is_observation.count_nonzero().item())

            value_logits, policy_logits = self.nets.prediction.raw_forward(
                latent,
            )
            value_target = self.nets.prediction.value_scale.get_target(step.value_target)
            losses.value += ml(cross, value_logits, value_target)
            losses.policy += ml(cross, policy_logits, step.target_policy)

            latent, reward_logits, turn_status_logits = self.nets.dynamics.raw_forward(
                latent,
                step.action_onehot,
            )
            reward_target = self.nets.dynamics.reward_scale.get_target(step.reward)
            losses.reward += ml(cross, reward_logits, reward_target)
            losses.turn += ml(cross, turn_status_logits, step.turn_status)

        counts.data = C.training.unroll_length * batch_size
        losses /= counts

        self.slaw.step(losses)
        weights = self.slaw.weights

        total_loss = losses.weighted_sum(weights)

        for k, loss in attrs.asdict(losses).items():
            tbs.add_scalar(f"loss/{k}", loss)
        for k, weight in weights.items():
            tbs.add_scalar(f"loss weight/{k}", weight)

        self.optimizer.zero_grad()
        total_loss.backward()  # type: ignore [no-untyped-call]
        self.optimizer.step()

        log.info(
            "Finished batch update, traj length {}, loss {:.3f}".format(
                len(batch),
                total_loss.item(),
            )
        )
