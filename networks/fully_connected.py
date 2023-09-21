import math
import itertools
from typing import Any, Type, Optional, cast
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from util import copy_type_signature

from .util import NecroReLu
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
        act_out: bool = False,
        hidden_depth: int = 2,
        width: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if width is None:
            width = int(input_width * 1.2)
        widths = [input_width] + [width] * hidden_depth + [output_width]
        self.act_out = act_out
        self.fcs, self.acts, self.norms = zip(
            *[
                (nn.Linear(a, b), NecroReLu(), nn.LayerNorm(a, elementwise_affine=False))
                for a, b in itertools.pairwise(widths)
            ]
        )
        for n, fc in enumerate(self.fcs):
            self.add_module(f"Fc{n}", fc)
        for n, act in enumerate(self.acts):
            self.add_module(f"Act{n}", act)
        for n, norm in enumerate(self.norms):
            self.add_module(f"norm{n}", norm)

    def forward(self, *inputs: Tensor) -> Tensor:
        x = torch.cat([i.flatten(1) for i in inputs], dim=1)
        last = self.fcs[-1]
        for fc, act, norm in zip(self.fcs, self.acts, self.norms):
            x = norm(x)
            x = fc(x)
            if fc is not last or self.act_out:
                x = act(x)
        return x

    @copy_type_signature(forward)  # provide typed __call__ interface
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return super().__call__(*args, **kwargs)


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
            act_out=True,
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
        self.act = NecroReLu()

    def forward(
        self,
        latent_in: Tensor,
        action_onehot: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        latent_out, reward, turn = self.fc_reshape(latent_in, action_onehot)
        latent_out = self.act(latent_out)
        return latent_in + latent_out, reward, turn
