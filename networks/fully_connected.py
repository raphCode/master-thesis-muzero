import math
import itertools
from typing import Any, Type, Optional, cast
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from util import copy_type_signature

from .bases import DynamicsNet, PredictionNet, RepresentationNet


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


class FlatReshaper(nn.Module):
    out_shapes: Sequence[Sequence[int]]

    def __init__(
        self,
        in_shapes: Sequence[Sequence[int]],
        out_shapes: Sequence[Sequence[int]],
        module: Type[nn.Module] = GenericFc,
        **kwargs: Any,
    ):
        super().__init__()
        self.out_shapes = out_shapes
        self.mod = module(
            input_width=sum(map(math.prod, in_shapes)),
            output_width=sum(map(math.prod, out_shapes)),
            **kwargs,
        )

    def forward(self, *inputs: Tensor) -> tuple[Tensor, ...]:
        def reshape_maybe(tensor: Tensor, shape: Sequence[int]) -> Tensor:
            if len(shape) > 1 and math.prod(shape) > 0:
                return tensor.reshape(-1, *shape)
            return tensor

        out_tensors = torch.split(
            self.mod(*inputs),
            list(map(math.prod, self.out_shapes)),
            dim=-1,
        )
        return tuple(
            reshape_maybe(tensor, shape)
            for tensor, shape in zip(out_tensors, self.out_shapes)
        )

    @copy_type_signature(forward)  # provide typed __call__ interface
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return super().__call__(*args, **kwargs)


class FcRepresentation(RepresentationNet):
    def __init__(self, latent_features: int, **kwargs: Any):
        from config import C

        super().__init__()
        self.fc_reshape = FlatReshaper(
            in_shapes=C.game.instance.observation_shapes,
            out_shapes=[[latent_features]],
            **kwargs,
        )

    def forward(
        self,
        *observations: Tensor,
    ) -> Tensor:
        return self.fc_reshape(*observations)[0]


class FcPrediction(PredictionNet):
    def __init__(self, **kwargs: Any):
        from config import C

        super().__init__()
        self.fc_reshape = FlatReshaper(
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
            self.fc_reshape(latent),
        )


class FcDynamics(DynamicsNet):
    def __init__(self, **kwargs: Any):
        from mcts import TurnStatus
        from config import C

        super().__init__()
        self.fc_reshape = FlatReshaper(
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
            self.fc_reshape(latent, action_onehot),
        )
