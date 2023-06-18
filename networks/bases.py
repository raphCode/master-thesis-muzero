import typing
import functools
from abc import ABC, abstractmethod
from typing import Any, Self, TypeVar, ParamSpec, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from attrs import define
from torch import Tensor

from util import copy_type_signature

P = ParamSpec("P")
R = TypeVar("R", bound=Tensor | tuple[Tensor, ...])


class NetBase(ABC, nn.Module):
    pass


# Custom network implementations should subclass the three Net classes below:

# beliefs are always Tensors, even when disabled.
# In this case, belief tensors are empty, i.e. they have at least one zero data dimension.
# eg, for 2D images, the tensor shape is often BxHxWxC. An empty tensor may have zero C.


class RepresentationNet(NetBase):
    # Observations (may include NumberPlayers, CurrentPlayer, TeamInfo) -> Latent

    @abstractmethod
    def forward(
        self,
        *observations: Tensor,
    ) -> Tensor:
        pass

    @copy_type_signature(forward)
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return super().__call__(*args, **kwargs)


class PredictionNet(NetBase):
    # Latent, Belief -> Value, Policy, CurrentPlayer

    @abstractmethod
    def forward(
        self,
        latent: Tensor,
        belief: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        pass

    @copy_type_signature(forward)
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return super().__call__(*args, **kwargs)


class DynamicsNet(NetBase):
    # Latent, Belief, Action -> Latent, Belief, Reward, IsTerminal

    @abstractmethod
    def forward(
        self,
        latent: Tensor,
        belief: Tensor,
        action_onehot: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        pass

    @copy_type_signature(forward)
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return super().__call__(*args, **kwargs)


class AutobatchWrapper(nn.Module):
    """
    Transparently adds/removes a singleton batch dimension when calling a wrapped module.
    """

    net: NetBase

    def __init__(self, network: NetBase, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.net = network

    def forward(self, *inputs: Tensor) -> Tensor | tuple[Tensor, ...]:
        unsqueeze = functools.partial(torch.unsqueeze, dim=0)
        squeeze = functools.partial(torch.squeeze, dim=0)

        result = self.net(*map(unsqueeze, inputs))
        if isinstance(result, tuple):
            return tuple(map(squeeze, result))
        return squeeze(result)


class NetContainer(
    ABC, nn.Module
):  # two containers, guard method self with mixin classes?
    r"""
    Container for the actual network implementation, providing single inference.
    Single inference means evaluating the network with unbatched data.

    The actual network implementation is hold in a submodule, and single inference is
    provided via AutobatchWrapper and a nested container:

           NetContainer
              /    \
          si /      \
            /        \
           v         |
     NetContainer    |
           |         |
       net |         | net
           |         |
           v         |
    AutobatchWrapper |
            \        |
         net \       |
              \      |
               v     v
            Implementation

    Using submodules is the preferred way in pytorch to reuse module functionality.
    It also allows to jit the model easily by mixing scripting and tracing to support
    dynamic control flow (conditional return of logits).
    """

    net: NetBase | AutobatchWrapper
    si: Self  # Actually not set on si containers

    def __init__(self, network: NetBase | AutobatchWrapper, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.net = network
        if not isinstance(network, AutobatchWrapper):
            net_type = typing.get_type_hints(self, globals())["net"]
            assert isinstance(network, net_type), (
                f"{net_type.__name__[:-3]} network must be subclass of {net_type} "
                f"(found instance of {type(network)})"
            )
            self.si = cast(Self, type(self))(AutobatchWrapper(network))

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        pass


class RepresentationNetContainer(NetContainer):
    net: RepresentationNet

    def forward(
        self,
        *observations: Tensor,
    ) -> Tensor:
        return self.net(*observations)

    @copy_type_signature(forward)  # export a typed __call__() interface
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return super().__call__(*args, **kwargs)


class PredictionNetContainer(NetContainer):
    net: PredictionNet

    def forward(
        self,
        latent: Tensor,
        belief: Tensor,
        logits: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        result = self.net(latent, belief)
        if logits:
            return result
        value, policy_log, player_log = result
        return value, F.softmax(policy_log, dim=-1), F.softmax(player_log, dim=-1)

    @copy_type_signature(forward)  # export a typed __call__() interface
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return super().__call__(*args, **kwargs)


class DynamicsNetContainer(NetContainer):
    net: DynamicsNet

    def forward(
        self,
        latent: Tensor,
        belief: Tensor,
        action_onehot: Tensor,
        logits: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        result = self.net(latent, belief, action_onehot)
        if logits:
            return result
        latent, belief, reward, terminal_log = result
        return latent, belief, reward, F.sigmoid(terminal_log)

    @copy_type_signature(forward)  # export a typed __call__() interface
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return super().__call__(*args, **kwargs)


@define(kw_only=True)
class Networks:
    representation: RepresentationNetContainer
    prediction: PredictionNetContainer
    dynamics: DynamicsNetContainer
    initial_latent: Tensor
    initial_belief: Tensor
