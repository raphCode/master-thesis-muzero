import math
import itertools
from typing import Any, Type, Optional, cast
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from networks.bases import (
    NetBase,
    DynamicsNet,
    NetContainer,
    PredictionNet,
    RepresentationNet,
    DynamicsNetContainer,
    PredictionNetContainer,
    RepresentationNetContainer,
)


class GenericFc(nn.Module):
    """
    Generic fully connected network implementation.
    """

    def __init__(
        self,
        input_width: int,
        output_width: int,
        *,
        hidden_depth: int = 2,
        width: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if width is None:
            width = input_width
        widths = [input_width] + [width] * hidden_depth + [output_width]
        self.fc_layers = [nn.Linear(a, b) for a, b in itertools.pairwise(widths)]
        for n, layer in enumerate(self.fc_layers):
            self.add_module(f"fc{n}", layer)

    def fc_forward(self, *inputs: Tensor) -> Tensor:
        x = torch.cat([i.flatten(1) for i in inputs], dim=1)
        for fc in self.fc_layers:
            x_in = x
            y = fc(x)  # type: Tensor
            x = F.relu(y)
            if fc.in_features == fc.out_features:
                x = x + x_in  # skip connection / ResNet
        return y


class FcBase(GenericFc, NetBase):
    container_type: Type[NetContainer]
    out_sizes: list[int]

    def __init__(self, **kwargs: Any):
        self.out_sizes = list(map(math.prod, self.container_type.out_shapes))
        super().__init__(
            input_width=sum(map(math.prod, self.container_type.in_shapes)),
            output_width=sum(self.out_sizes),
            **kwargs,
        )

    def reshape_forward(self, *inputs: Tensor) -> tuple[Tensor, ...]:
        def reshape_maybe(tensor: Tensor, shape: Sequence[int]) -> Tensor:
            if len(shape) > 1 and math.prod(shape) > 0:
                return tensor.reshape(-1, *shape)
            return tensor

        out_tensors = torch.split(self.fc_forward(*inputs), self.out_sizes, dim=-1)
        return tuple(
            reshape_maybe(tensor, shape)
            for tensor, shape in zip(out_tensors, self.container_type.out_shapes)
        )


class FcRepresentation(FcBase, RepresentationNet):
    container_type: Type[NetContainer] = RepresentationNetContainer

    def forward(
        self,
        *observations: Tensor,
    ) -> Tensor:
        return self.reshape_forward(*observations)[0]


class FcPrediction(FcBase, PredictionNet):
    container_type: Type[NetContainer] = PredictionNetContainer

    def forward(self, latent: Tensor, belief: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return cast(
            tuple[Tensor, Tensor, Tensor],
            self.reshape_forward(latent, belief),
        )


class FcDynamics(FcBase, DynamicsNet):
    container_type: Type[NetContainer] = DynamicsNetContainer

    def forward(
        self,
        latent: Tensor,
        belief: Tensor,
        action_onehot: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return cast(
            tuple[Tensor, Tensor, Tensor, Tensor],
            self.reshape_forward(latent, belief, action_onehot),
        )
