import math
import itertools
from typing import Any, Callable, Optional, cast
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from util import copy_type_signature

from .util import NoOp, ModuleFactory
from .bases import DynamicsNet, PredictionNet, RepresentationNet


class FcImpl(nn.Module):
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
        first_layer_pre_activation: bool = True,
        use_weight_norm: bool = False,
        normalisation: Callable[..., nn.Module] = NoOp,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.first_layer_pre_activation = first_layer_pre_activation
        if width is None:
            width = input_width
        widths = [input_width] + [width] * hidden_depth + [output_width]
        fc_factory = ModuleFactory(self, nn.Linear, "fc")
        self.fc_layers = [fc_factory(a, b) for a, b in itertools.pairwise(widths)]
        if use_weight_norm:
            for fc in self.fc_layers:
                nn.utils.parametrizations.weight_norm(fc)
        norm_factory = ModuleFactory(self, normalisation, "norm")
        self.norms = [norm_factory(a) for a in widths[:-1]]

    def fc_forward(self, *inputs: Tensor) -> Tensor:
        x = torch.cat([i.flatten(1) for i in inputs], dim=1)
        first = self.fc_layers[0]
        last = self.fc_layers[-1]
        for fc, norm in zip(self.fc_layers, self.norms):
            skip = x
            if not first or self.first_layer_pre_activation:
                x = norm(x)
                x = F.relu(x)
            x = fc(x)
            if fc is not last and fc.in_features == fc.out_features:
                x = x + skip  # skip connection / ResNet
        return x


class GenericFc(FcImpl):
    def forward(self, *inputs: Tensor) -> Tensor:
        return self.fc_forward(*inputs)

    @copy_type_signature(forward)
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return super().__call__(*args, **kwargs)


class FcReshaperImpl(FcImpl):
    out_shapes: Sequence[Sequence[int]]

    def __init__(
        self,
        in_shapes: Sequence[Sequence[int]],
        out_shapes: Sequence[Sequence[int]],
        **kwargs: Any,
    ):
        self.out_shapes = out_shapes
        super().__init__(
            input_width=sum(map(math.prod, in_shapes)),
            output_width=sum(map(math.prod, out_shapes)),
            **kwargs,
        )

    def reshape_forward(self, *inputs: Tensor) -> tuple[Tensor, ...]:
        def reshape_maybe(tensor: Tensor, shape: Sequence[int]) -> Tensor:
            if len(shape) > 1 and math.prod(shape) > 0:
                return tensor.reshape(-1, *shape)
            return tensor

        out_tensors = torch.split(
            self.fc_forward(*inputs),
            list(map(math.prod, self.out_shapes)),
            dim=-1,
        )
        return tuple(
            reshape_maybe(tensor, shape)
            for tensor, shape in zip(out_tensors, self.out_shapes)
        )


class FcReshaper(FcReshaperImpl):
    def forward(self, *inputs: Tensor) -> tuple[Tensor, ...]:
        return self.reshape_forward(*inputs)

    @copy_type_signature(forward)
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return super().__call__(*args, **kwargs)


class FcRepresentation(FcReshaperImpl, RepresentationNet):
    def __init__(self, latent_features: int, **kwargs: Any):
        from config import C

        super().__init__(
            in_shapes=C.game.instance.observation_shapes,
            out_shapes=[[latent_features]],
            **(dict(first_layer_pre_activation=False) | kwargs),
        )

    def forward(
        self,
        *observations: Tensor,
    ) -> Tensor:
        return self.reshape_forward(*observations)[0]


class FcPrediction(FcReshaperImpl, PredictionNet):
    def __init__(self, **kwargs: Any):
        from config import C

        super().__init__(
            in_shapes=[
                C.networks.latent_shape,
            ],
            out_shapes=[
                [C.networks.scalar_support_size, C.game.instance.max_num_players],
                [C.game.instance.max_num_actions],
            ],
            **kwargs,
        )

    def forward(self, latent: Tensor) -> tuple[Tensor, Tensor]:
        return cast(
            tuple[Tensor, Tensor],
            self.reshape_forward(latent),
        )


class FcDynamics(FcReshaperImpl, DynamicsNet):
    def __init__(self, **kwargs: Any):
        from mcts import TurnStatus
        from config import C

        super().__init__(
            in_shapes=[
                C.networks.latent_shape,
                [C.game.instance.max_num_actions],
            ],
            out_shapes=[
                C.networks.latent_shape,
                [C.networks.scalar_support_size, C.game.instance.max_num_players],
                [C.game.instance.max_num_players + len(TurnStatus)],
            ],
            **kwargs,
        )

    def forward(
        self,
        latent: Tensor,
        action_onehot: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return cast(
            tuple[Tensor, Tensor, Tensor],
            self.reshape_forward(latent, action_onehot),
        )
