from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
import torch.nn as nn
from attrs import define
from torch import Tensor

from util import optional_map, copy_type_signature


def single_inference(
    net: RepresentationNet | PredictionNet | DynamicsNet,
    inputs: tuple[Tensor, ...],
    /,
    **kwargs: Any,
) -> Tensor | tuple[Optional[Tensor], ...]:
    """
    Adds/removes batch dimension on network in/outputs.
    """
    unsqueeze = optional_map(functools.partial(torch.unsqueeze, dim=0))
    squeeze = functools.partial(torch.squeeze, dim=0)

    result = net(*map(unsqueeze, inputs), **kwargs)  # type: ignore [arg-type]
    if isinstance(result, tuple):
        return tuple(map(optional_map(squeeze), result))
    return squeeze(result)


class RepresentationNet(ABC, nn.Module):
    # Observations (may include NumberPlayers, CurrentPlayer, TeamInfo) -> Latent
    @abstractmethod
    def forward(
        self,
        *observations: Tensor,
    ) -> Tensor:
        pass

    @copy_type_signature(forward)
    def si(self, *inputs: Tensor, **kwargs: Any) -> Any:
        return single_inference(self, inputs, **kwargs)

    @copy_type_signature(forward)
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return super().__call__(*args, **kwargs)


class PredictionNet(ABC, nn.Module):
    # Latent, Belief -> Value, Policy, CurrentPlayer
    @abstractmethod
    def forward(
        self,
        latent: Tensor,
        belief: Optional[Tensor],
        logits: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        pass

    @copy_type_signature(forward)
    def si(self, *inputs: Tensor, **kwargs: Any) -> Any:
        return single_inference(self, inputs, **kwargs)

    @copy_type_signature(forward)
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return super().__call__(*args, **kwargs)


class DynamicsNet(ABC, nn.Module):
    # Latent, Belief, Action -> Latent, Belief, Reward, IsTerminal
    @abstractmethod
    def forward(
        self,
        latent: Tensor,
        belief: Optional[Tensor],
        action_onehot: Tensor,
        logits: bool = False,
    ) -> tuple[Tensor, Optional[Tensor], Tensor, Tensor]:
        pass

    @copy_type_signature(forward)
    def si(self, *inputs: Tensor, **kwargs: Any) -> Any:
        return single_inference(self, inputs, **kwargs)

    @copy_type_signature(forward)
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return super().__call__(*args, **kwargs)


@define(kw_only=True)
class Networks:
    representation: RepresentationNet
    prediction: PredictionNet
    dynamics: DynamicsNet
    initial_latent: Tensor
    initial_belief: Optional[Tensor]
