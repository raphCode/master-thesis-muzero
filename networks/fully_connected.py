import math
from util import copy_type_signature
import itertools
from typing import Any, Optional, cast, Callable
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .bases import NetBase, DynamicsNet, PredictionNet, RepresentationNet

from .util import ModuleFactory, NoOp


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
        raw_out: bool = True,
        normalisation: Callable[..., nn.Module] = NoOp,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.raw_out = raw_out
        if width is None:
            width = int(min(input_width, output_width))
        widths = [input_width] + [width] * hidden_depth + [output_width]
        fc_factory = ModuleFactory(self, nn.Linear, "fc")
        self.fc_layers = [fc_factory(a, b) for a, b in itertools.pairwise(widths)]
        norm_factory = ModuleFactory(self, normalisation, "norm")
        self.norms = [norm_factory(b) for b in widths[1:]]

    def fc_forward(self, *inputs: Tensor) -> Tensor:
        x = torch.cat([i.flatten(1) for i in inputs], dim=1)
        *_, last = self.fc_layers
        for fc, norm in zip(self.fc_layers, self.norms):
            skip = x
            x = fc(x)
            if fc is last and self.raw_out:
                return x
            x = norm(x)
            x = F.relu(x)
            if fc.in_features == fc.out_features:
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
    def forward(self, *inputs: Tensor) -> Tensor:
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
            raw_out=False,
            **kwargs,
        )
        self.norm = nn.BatchNorm1d(latent_features)

    def forward(
        self,
        *observations: Tensor,
    ) -> Tensor:
        return self.norm(self.reshape_forward(*observations)[0])


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
        self.norm = nn.BatchNorm1d(C.networks.latent_shape[0])

    def forward(
        self,
        latent: Tensor,
        action_onehot: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        latent, reward, turn = self.reshape_forward(latent, action_onehot)
        return self.norm(latent), reward, turn
