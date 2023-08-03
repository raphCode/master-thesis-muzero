from __future__ import annotations

import logging
import operator
import functools
from typing import TYPE_CHECKING, Callable, Optional, cast

import attrs
import numpy as np
import scipy  # type: ignore
import torch
from attrs import Factory, define
from torch import Tensor, nn

from util import RingBuffer, ndarr_f64
from config import C
from config.schema import LossWeights

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


class SLAW:
    """
    Implements automatic loss weighting as described in:
    "Scaled Loss Approximate Weighting for Efficient Multi-Task Learning"
    https://arxiv.org/abs/2109.08218
    """

    beta: float
    names: tuple[str, ...]  # names of loss components and weights, in order
    a: ndarr_f64
    b: ndarr_f64
    w: ndarr_f64

    def __init__(self, mavg_beta: float) -> None:
        self.beta = mavg_beta
        self.names = tuple(
            f.name for f in attrs.fields(LossWeights) #    if f.name != "latent"
        )
        n = len(self.names)
        self.a = np.zeros(n)
        self.b = np.zeros(n)

    def step(self, losses: Losses, tbs: TBStepLogger) -> None:
        """
        Update internal loss weights based on the current training step's losses.
        """
        get_fields = operator.attrgetter(*self.names)
        l = np.array([t.item() for t in get_fields(losses)])
        n = len(self.names)
        self.a = self.beta * self.a + (1 - self.beta) * l**2
        self.b = self.beta * self.b + (1 - self.beta) * l
        s = np.maximum(1e-5, np.sqrt(self.a - self.b**2))
        self.w = (n / s) / np.sum(1 / s)

        tbs.add_scalar("slaw: w/std", np.std(self.w))  # type: ignore [arg-type]
        tbs.add_scalar("slaw: s/mean", np.mean(s))
        tbs.add_scalar("slaw: s/geomean", scipy.stats.gmean(s))
        tbs.add_scalar("slaw: s/std", np.std(s))
        tbs.add_scalar("slaw: s/std/mean", np.std(s) / np.mean(s))
        for name, ss in zip(self.names, s):
            tbs.add_scalar("slaw: s/" + name, ss)

    @property
    def weights(self) -> LossWeights:
        """
        Return the calculated loss weights.
        """
        return LossWeights(**dict(zip(self.names, self.w)))#   ,latent=0)


class MyReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def step(self, metrics: float, epoch: Optional[int] = None) -> None:
        if self.in_cooldown:
            self.best = self.mode_worse
        return super().step(metrics)


class Trainer:
    def __init__(self, nets: Networks):
        self.nets = nets
        lrs = C.training.learning_rates
        self.slaw = SLAW(0.999)
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
        self.loss_history = RingBuffer[float](20)
        self.lr_scheduler = MyReduceLROnPlateau(
            self.optimizer, factor=0.5, cooldown=1000, verbose=True, patience=100,
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

        self.nets.train()
        pdist = nn.PairwiseDistance(p=C.training.latent_dist_pnorm)
        cross = nn.CrossEntropyLoss(reduction="none")
        mse = nn.MSELoss(reduction="none")
        huber = nn.HuberLoss(reduction="none")
        cos = functools.partial(
            nn.CosineEmbeddingLoss(reduction="none"), target=torch.ones(1)
        )

        losses = Losses()
        counts = LossCounts()

        first = batch[0]
        latent = first.latent
        latent[first.is_initial] = self.nets.initial_latent

        for n,step in enumerate(batch):
            if step.is_observation.any():
                obs_latent = self.nets.representation(*step.observations)
                if step is first:
                    latent[step.is_observation] = obs_latent[step.is_observation]
                else:
                    losses.latent += ml(
                        cos,
                        latent.flatten(start_dim=1),
                        obs_latent.flatten(start_dim=1),
                        mask=step.is_observation,
                    )
                    counts.latent += cast(int, step.is_observation.count_nonzero().item())

            tbs.add_histogram(f"latent/unroll {n}", latent.clamp(-10, 10))
            counts.data += cast(int, step.is_data.count_nonzero().item())

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

        losses /= counts

        self.slaw.step(losses, tbs)
        weights = self.slaw.weights
        #weights=C.training.loss_weights


        for k, loss in attrs.asdict(losses).items():
            tbs.add_scalar(f"loss/{k}", loss)
        for k, weight in attrs.asdict(weights).items():
            tbs.add_scalar(f"loss weight/{k}", weight)

        total_loss = losses.weighted_sum(weights)
        self.loss_history.append(total_loss.item())
        self.lr_scheduler.step(np.mean(self.loss_history))  # type: ignore [call-overload]
        for n, param_group in enumerate(self.optimizer.param_groups):
            # TODO: get lr from scheduler
            tbs.add_scalar(f"lr/param group {n}", param_group["lr"])

        if self.loss_history.fullness == 1:
            old, new = np.split(np.array(self.loss_history), 2)
            loss_diff = new.mean() - old.mean()
            tbs.add_scalar("loss/diff", loss_diff)
        tbs.add_scalar("loss/mean", np.mean(self.loss_history))  # type: ignore [call-overload]

        tbs.add_scalar("loss/total", total_loss)
        tbs.add_scalar("loss/unweighted sum", sum(attrs.astuple(losses)))

        self.optimizer.zero_grad()
        total_loss.backward()  # type: ignore [no-untyped-call]
        self.optimizer.step()

        log.info(
            "Finished batch update, traj length {}, loss {:.5f}".format(
                len(batch),
                total_loss.item(),
            )
        )
