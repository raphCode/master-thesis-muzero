from __future__ import annotations

import typing
import logging
import functools
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, cast
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from util import copy_type_signature

from .rescaler import Rescaler

if TYPE_CHECKING:
    from torch import Tensor

    from .bases import NetBase, DynamicsNet, PredictionNet, RepresentationNet

log = logging.getLogger(__name__)


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
        import networks

        namespace = vars(networks.bases) | vars(networks.rescaler)
        net_type = typing.get_type_hints(self, namespace)["net"]
        assert isinstance(network, net_type), (
            f"{net_type.__name__[:-3]} network must be subclass of {net_type} "
            f"(found instance of {type(network)})"
        )

    @classmethod
    @abstractmethod
    def _input_shapes(cls) -> Sequence[Sequence[int]]:
        """
        Shapes of the network inputs. Used for jit tracing.
        """
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
        from config import C

        @functools.cache
        def example_inputs(unbatched: bool = False) -> list[Tensor]:
            def example_tensor(shape: Sequence[int]) -> Tensor:
                t = torch.zeros(*shape)
                if unbatched:
                    return t
                return t.expand(C.training.batch_size, *shape)

            return list(map(example_tensor, self._input_shapes()))

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

        node_count = sum(1 for _ in jit_mod.inlined_graph.nodes())
        log.info(f"Jit traced {type(self).__name__} ({node_count} Nodes)")
        return jit_mod


class RepresentationNetContainer(NetContainer):
    net: RepresentationNet

    @classmethod
    def _input_shapes(cls) -> Sequence[Sequence[int]]:
        from config import C

        return C.game.instance.observation_shapes

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
        from config import C

        self.value_scale = Rescaler(C.networks.scalar_support_size)

    @classmethod
    def _input_shapes(cls) -> Sequence[Sequence[int]]:
        from config import C

        return [C.networks.latent_shape]

    def forward(
        self,
        latent: Tensor,
    ) -> tuple[Tensor, Tensor]:
        value_log, policy_log = self.net(latent)
        return self.value_scale(value_log), F.softmax(policy_log, dim=1)

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
        from config import C

        self.reward_scale = Rescaler(C.networks.scalar_support_size)

    @classmethod
    def _input_shapes(cls) -> Sequence[Sequence[int]]:
        from config import C

        return (
            C.networks.latent_shape,
            [C.game.instance.max_num_actions],
        )

    def forward(
        self,
        latent: Tensor,
        action_onehot: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        latent, reward_log, turn_status_log = self.net(latent, action_onehot)
        return (
            latent,
            self.reward_scale(reward_log),
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
