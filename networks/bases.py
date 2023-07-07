import typing
import functools
from abc import ABC, abstractmethod
from typing import Any, TypeVar, Protocol, ParamSpec, TypeAlias, cast
from collections.abc import Sequence

import attrs
import torch
import torch.nn as nn
import torch.nn.functional as F
from attrs import define
from torch import Tensor

from mcts import TurnStatus
from util import copy_type_signature
from config import C

from .rescaler import Rescaler

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

    # method overrides are to provide properly typed function signatures:
    @copy_type_signature(forward)
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

    # method overrides are to provide properly typed function signatures:
    @copy_type_signature(forward)
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

    # method overrides are to provide properly typed function signatures:
    @copy_type_signature(forward)
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return super().__call__(*args, **kwargs)


def make_shapes(*shapes: int | Sequence[int]) -> Shapes:
    """
    Create a list of shape tuples, promoting single integer shapes to a tuple.
    """

    def int2shape(x: int | Sequence[int]) -> Sequence[int]:
        return [x] if isinstance(x, int) else x

    return list(map(tuple[int], map(int2shape, shapes)))


def autobatch(module: nn.Module, *inputs: Tensor) -> Tensor | tuple[Tensor, ...]:
    """
    Adds/removes a singleton batch dimension when calling a module.
    """
    unsqueeze = functools.partial(torch.unsqueeze, dim=0)
    squeeze = functools.partial(torch.squeeze, dim=0)

    result = module(*map(unsqueeze, inputs))
    if isinstance(result, tuple):
        return tuple(map(squeeze, result))
    return squeeze(result)


class NetContainer(ABC, nn.Module):
    """
    Wraps actual network implementation, provides value rescaling and single inference.
    Single inference means evaluating the network with unbatched data.
    """

    net: NetBase

    def __init__(self, network: NetBase, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.net = network
        net_type = typing.get_type_hints(self, globals())["net"]
        assert isinstance(network, net_type), (
            f"{net_type.__name__[:-3]} network must be subclass of {net_type} "
            f"(found instance of {type(network)})"
        )

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

    @abstractmethod
    def si(self, *args: Any, **kwargs: Any) -> Any:
        """
        Single inference: in/output tensors without batch dimension
        """
        pass

    def raw_forward(self, *inputs: Any, **kwargs: Any) -> Any:
        return self.net(*inputs)

    def jit(self) -> torch.jit.TopLevelTracedModule:
        @functools.cache
        def example_inputs(unbatched: bool = False) -> list[Tensor]:
            def example_tensor(shape: Sequence[int]) -> Tensor:
                t = torch.zeros(*shape)
                if unbatched:
                    return t
                return t.expand(C.training.batch_size, *shape)

            return list(map(example_tensor, self.in_shapes))

        for name, mod in self.named_modules():
            if mod is not self and hasattr(mod, "jit") and callable(mod.jit):
                setattr(self, name, mod.jit())
        return cast(
            torch.jit.TopLevelTracedModule,
            torch.jit.trace_module(  # type: ignore [no-untyped-call]
                self,
                dict(
                    forward=example_inputs(),
                    raw_forward=example_inputs(),
                    si=example_inputs(unbatched=True),
                ),
            ),
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

    # method overrides are to provide properly typed function signatures:
    @copy_type_signature(forward)
    def si(self, *inputs: Tensor) -> Any:
        return autobatch(self, *inputs)

    @copy_type_signature(forward)
    def raw_forward(self, *inputs: Tensor) -> Any:
        return self.net(*inputs)

    @copy_type_signature(forward)
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return super().__call__(*args, **kwargs)


class PredictionNetContainer(NetContainer):
    net: PredictionNet
    value_scale: Rescaler

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.value_scale = Rescaler()

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
    ) -> tuple[Tensor, Tensor]:
        value, policy_log = self.net(latent, belief)
        return self.value_scale.rescale(value), F.softmax(policy_log, dim=1)

    # method overrides are to provide properly typed function signatures:
    @copy_type_signature(forward)
    def si(self, *inputs: Tensor) -> Any:
        return autobatch(self, *inputs)

    @copy_type_signature(forward)
    def raw_forward(self, *inputs: Tensor) -> Any:
        return self.net(*inputs)

    @copy_type_signature(forward)
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return super().__call__(*args, **kwargs)


class DynamicsNetContainer(NetContainer):
    net: DynamicsNet
    reward_scale: Rescaler

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.reward_scale = Rescaler()

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
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        latent, belief, reward, turn_status_log = self.net(latent, belief, action_onehot)
        return (
            latent,
            belief,
            self.reward_scale.rescale(reward),
            F.softmax(turn_status_log, dim=1),
        )

    # method overrides are to provide properly typed function signatures:
    @copy_type_signature(forward)
    def si(self, *inputs: Tensor) -> Any:
        return autobatch(self, *inputs)

    @copy_type_signature(forward)
    def raw_forward(self, *inputs: Tensor) -> Any:
        return self.net(*inputs)

    @copy_type_signature(forward)
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return super().__call__(*args, **kwargs)


class ProvidesBounds(Protocol):
    @property
    def value_bounds(self) -> tuple[float, float]:
        ...

    @property
    def reward_bounds(self) -> tuple[float, float]:
        ...


@define(kw_only=True)
class Networks:
    representation: RepresentationNetContainer
    prediction: PredictionNetContainer
    dynamics: DynamicsNetContainer
    initial_latent: Tensor
    initial_belief: Tensor

    def jit(self) -> None:
        for name, item in attrs.asdict(self).items():
            if isinstance(item, NetContainer):
                setattr(self, name, item.jit())

    def update_rescalers(self, b: ProvidesBounds) -> None:
        self.prediction.value_scale.update_bounds(b.value_bounds)
        self.dynamics.reward_scale.update_bounds(b.reward_bounds)
