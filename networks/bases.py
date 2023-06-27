import typing
import functools
from abc import ABC, abstractmethod
from typing import Any, Self, TypeVar, ParamSpec, TypeAlias, cast
from collections.abc import Sequence

import attrs
import torch
import torch.nn as nn
import torch.nn.functional as F
from attrs import define
from torch import Tensor

from mcts import TurnStatus
from util import copy_type_signature, hide_type_annotations
from config import C

P = ParamSpec("P")
R = TypeVar("R", bound=Tensor | tuple[Tensor, ...])


Shapes: TypeAlias = list[tuple[int, ...]]
ShapesInfo: TypeAlias = Sequence[int | Sequence[int]]


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

    @copy_type_signature(forward)  # export a typed __call__() interface
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return super().__call__(*args, **kwargs)


class PredictionNet(NetBase):
    # Latent, Belief -> Value, Policy

    @abstractmethod
    def forward(
        self,
        latent: Tensor,
        belief: Tensor,
    ) -> tuple[Tensor, Tensor]:
        pass

    @copy_type_signature(forward)  # export a typed __call__() interface
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return super().__call__(*args, **kwargs)


class DynamicsNet(NetBase):
    # Latent, Belief, Action -> Latent, Belief, Reward, TurnStatus

    @abstractmethod
    def forward(
        self,
        latent: Tensor,
        belief: Tensor,
        action_onehot: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        pass

    @copy_type_signature(forward)  # export a typed __call__() interface
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return super().__call__(*args, **kwargs)


def make_shapes(*shapes: int | Sequence[int]) -> Shapes:
    """
    Create a list of shape tuples, promoting single integer shapes to a tuple.
    """

    def int2shape(x: int | Sequence[int]) -> Sequence[int]:
        return [x] if isinstance(x, int) else x

    return list(map(tuple[int], map(int2shape, shapes)))


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

    @classmethod  # type: ignore [misc]
    @property
    @cast("functools._lru_cache_wrapper[Shapes]", functools.cache)
    def in_shapes(cls) -> Shapes:
        """
        Input tensor shapes according to the global config (without batch dim).
        """
        return make_shapes(*cls._in_shape_info())

    @classmethod  # type: ignore [misc]
    @property
    @cast("functools._lru_cache_wrapper[Shapes]", functools.cache)
    def out_shapes(cls) -> Shapes:
        """
        Output tensor shapes according to the global config (without batch dim).
        """
        return make_shapes(*cls._out_shape_info())

    @classmethod
    @abstractmethod
    def _in_shape_info(cls) -> ShapesInfo:
        pass

    @classmethod
    @abstractmethod
    def _out_shape_info(cls) -> ShapesInfo:
        pass

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def jit(
        self, jit_container: bool = True
    ) -> Self | torch.jit.TopLevelTracedModule | torch.jit.ScriptModule:
        def example_tensor(shape: Sequence[int]) -> Tensor:
            t = torch.zeros(*shape)
            if isinstance(self.net, AutobatchWrapper):
                return t
            return t.expand(C.training.batch_size, *shape)

        example_inputs = list(map(example_tensor, self.in_shapes))
        traced_net = torch.jit.trace(  # type: ignore [no-untyped-call]
            self.net, example_inputs
        )
        self.net = traced_net
        if hasattr(self, "si"):
            self.si.net = AutobatchWrapper(traced_net)
            self.si = self.si.jit(jit_container)  # type: ignore [assignment]
        if not jit_container:
            return self
        if isinstance(self, RepresentationNetContainer):
            # forward() has varargs, this can only be traced
            return cast(
                torch.jit.TopLevelTracedModule,
                torch.jit.trace(self, example_inputs),  # type: ignore [no-untyped-call]
            )
        else:
            # forward() has dynamic control flow (logits), use scripting
            with hide_type_annotations(NetContainer, "si", "net"):
                # These class type annotations are not really instance attributes but
                # submodules, covered by nn.Module's __getattr__ lookup
                # We need them for mypy, but they confuse torch.jit.script,
                # so temporarily delete them
                with hide_type_annotations(type(self), "si", "net"):
                    return cast(
                        torch.jit.ScriptModule,
                        torch.jit.script(self),
                    )


class RepresentationNetContainer(NetContainer):
    net: RepresentationNet

    @classmethod
    def _in_shape_info(cls) -> ShapesInfo:
        return list(C.game.instance.observation_shapes)

    @classmethod
    def _out_shape_info(cls) -> ShapesInfo:
        return [C.networks.latent_shape]

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

    @classmethod
    def _in_shape_info(cls) -> ShapesInfo:
        return C.networks.latent_shape, C.networks.belief_shape

    @classmethod
    def _out_shape_info(cls) -> ShapesInfo:
        return (
            1,
            C.game.instance.max_num_actions,
        )

    def forward(
        self,
        latent: Tensor,
        belief: Tensor,
        logits: bool = False,
    ) -> tuple[Tensor, Tensor]:
        result = self.net(latent, belief)
        if logits:
            return result
        value, policy_log = result
        return value, F.softmax(policy_log, dim=-1)

    @copy_type_signature(forward)  # export a typed __call__() interface
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return super().__call__(*args, **kwargs)


class DynamicsNetContainer(NetContainer):
    net: DynamicsNet

    @classmethod
    def _in_shape_info(cls) -> ShapesInfo:
        return (
            C.networks.latent_shape,
            C.networks.belief_shape,
            C.game.instance.max_num_actions,
        )

    @classmethod
    def _out_shape_info(cls) -> ShapesInfo:
        return (
            C.networks.latent_shape,
            C.networks.belief_shape,
            1,
            C.game.instance.max_num_players + len(TurnStatus),
        )

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
        latent, belief, reward, turn_status_log = result
        return latent, belief, reward, F.softmax(turn_status_log, dim=-1)

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

    def jit(self, jit_container: bool = True) -> None:
        for name, item in attrs.asdict(self).items():
            if isinstance(item, NetContainer):
                setattr(self, name, item.jit(jit_container))
