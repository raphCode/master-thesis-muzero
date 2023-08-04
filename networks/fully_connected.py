import math
import itertools
from typing import Any, Type, Optional, cast
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from util import copy_type_signature

from .bases import DynamicsNet, PredictionNet, RepresentationNet


def raph_relu(x: Tensor) -> Tensor:
    forward = F.relu(x)
    backward = F.leaky_relu(x)
    return backward + (forward - backward).detach()


class RaphRELU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return raph_relu(x)


class BasicBlock(nn.Module):
    def __init__(
        self,
        input_width: int,
        output_width: Optional[int] = None,
        norm: Type[nn.Module] = nn.BatchNorm1d,
        # activation: Type[nn.Module] = partial(nn.CELU, alpha=0.1),
        activation: Type[nn.Module] = RaphRELU,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.act = activation()
        self.norm = norm(input_width)
        self.fc = nn.Linear(input_width, output_width or input_width)

    def forward(self, x: Tensor) -> Tensor:
        x = self.act(x)
        x = self.norm(x)
        x = self.fc(x)
        return x

    @copy_type_signature(forward)  # provide typed __call__ interface
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return super().__call__(*args, **kwargs)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        layers: Sequence[nn.Module],
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, x + self.block(x))

    @copy_type_signature(forward)  # provide typed __call__ interface
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return super().__call__(*args, **kwargs)


"""

class FcResidualXBlock(nn.Module):
    def __init__(
        self,
        signal_width: int,
        depth: int = 2,
        width: Optional[int] = None,
        side_input_width: int = 0,
        side_output_width: int = 0,
        activation: Type[nn.Module] = nn.CELU,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        assert depth > 0
        if width is None:
            width = signal_width + max(side_input_width, side_output_width)
        widths = (
            [signal_width + side_input_width]
            + [width] * (depth - 1)
            + [signal_width + side_output_width]
        )
        layers = [nn.Linear(a, b) for a, b in itertools.pairwise(widths)]
        activations = [nn.CELU() for _ in range(depth)]
        self.block = nn.Sequential(*itertoolz.interleave([activations, layers]))
        self.splits = [signal_width, side_output_width]

    def forward(
        self, x: Tensor, side_in: Optional[Tensor] = None
    ) -> Tensor | tuple[Tensor, Tensor]:
        skip = x
        if side_in is not None:
            x = torch.cat([x, side_in], dim=1)
        x = self.block(x)
        residual, side_out = torch.split(x, self.splits, dim=1)
        x = skip + residual
        if side_out.numel() > 0:
            return x, side_out
        return x

    @copy_type_signature(forward)  # provide typed __call__ interface
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return super().__call__(*args, **kwargs)


class XResidualBlock(nn.Module):
    def __init__(
        self,
        layers: Sequence[FcResidualXBlock],
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor, side_input: Tensor) -> tuple[Tensor, Tensor]:
        n = len(x)
        skip = x
        x = torch.cat([x, side_input], dim=1)
        x = self.block(x)
        residual, side_output = torch.split(x, [n, len(x) - n], dim=1)
        return x + residual, side_output

    @copy_type_signature(forward)  # provide typed __call__ interface
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return super().__call__(*args, **kwargs)


"""
"""
class ResNet(nn.Module):
    def __init__(
        self,
        width: int,
        block_depth: int, 
        num_blocks:int
        blocks:Sequence[nn.Module],
        skip_every:int,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.blocks=nn.ModuleList(blocks)
        self.skip_every=skip_every

    def forward(
        self, x: Tensor
    ) -> Tensor :
        for n, block in enumerate(self.blocks):
            if n % self.skip_every == 0:
                skip =x
            x = block(x)
            if (n+1)%self.skip_every ==0 and skip.shape == x.shape:
                x = x + skip
        return x

    @copy_type_signature(forward)  # provide typed __call__ interface
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return super().__call__(*args, **kwargs)
    
"""


def skip_connections(layers: Sequence[nn.Module], x: Tensor) -> Tensor:
    skip = None
    for n, layer in enumerate(layers):
        if n % 2 == 0:
            skip = x
        if n > 0:
            x = F.relu(x)  # pre-activation
        x = layer(x)
        if n % 2 == 0 and skip is not None and skip.shape == x.shape:
            x = x + skip
    return x


class GenericFc(nn.Module):
    """
    Generic fully connected network implementation.
    """

    def __init__(
        self,
        input_width: int,
        output_width: int,
        *,
        depth: int = 1,
        width: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if width is None:
            width = input_width
        widths = [input_width] + [width] * (depth - 1) + [output_width]
        self.fc_layers = [nn.Linear(a, b) for a, b in itertools.pairwise(widths)]
        for n, layer in enumerate(self.fc_layers):
            self.add_module(f"fc{n}", layer)

    def forward(self, *inputs: Tensor) -> Tensor:
        x = torch.cat([i.flatten(1) for i in inputs], dim=1)
        return skip_connections(self.fc_layers, x)


class SplitReshape(nn.Module):
    def __init__(
        self,
        shapes: Sequence[Sequence[int]],
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.shapes = shapes

    def forward(self, x: Tensor) -> Tensor | list[Tensor]:
        outputs = torch.split(x, list(map(math.prod, self.shapes)), dim=1)
        outputs = [t.reshape(t.shape[0], *s) for t, s in zip(outputs, self.shapes)]
        if len(outputs) == 1:
            return outputs[0]
        return outputs

    @copy_type_signature(forward)  # provide typed __call__ interface
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return super().__call__(*args, **kwargs)


class FcSplitReshapeHead(nn.Module):
    def __init__(
        self,
        input_width: int,
        output_shapes: Sequence[Sequence[int]],
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        output_width = sum(map(math.prod, output_shapes))
        self.block = BasicBlock(input_width, output_width)
        self.split_reshape = SplitReshape(output_shapes)

    def forward(self, x: Tensor) -> Tensor | list[Tensor]:
        return self.split_reshape(self.block(x))

    @copy_type_signature(forward)  # provide typed __call__ interface
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return super().__call__(*args, **kwargs)


class FcSplitOutputs(nn.Module):
    """
    Fully connected network that supports multiple in/outputs of various tensor shapes.
    Inputs are flattened, concatenated and processed by the fully connected network.
    The output is split and reshaped into multiple Tensors.
    """

    out_shapes: Sequence[Sequence[int]]

    def __init__(
        self,
        in_shapes: Sequence[Sequence[int]],
        out_shapes: Sequence[Sequence[int]],
        **kwargs: Any,
    ):
        super().__init__()
        self.out_shapes = out_shapes
        self.fc = GenericFc(
            input_width=sum(map(math.prod, in_shapes)),
            output_width=sum(map(math.prod, out_shapes)),
            **kwargs,
        )

    def forward(self, *inputs: Tensor) -> tuple[Tensor, ...]:
        def reshape_maybe(tensor: Tensor, shape: Sequence[int]) -> Tensor:
            return tensor.reshape(tensor.shape[0], *shape)

        out_tensors = torch.split(
            self.fc(*inputs),
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
    fc_split: FcSplitOutputs

    def __init__(self, **kwargs: Any):
        super().__init__()
        from config import C

        self.fc_split = FcSplitOutputs(
            in_shapes=C.game.instance.observation_shapes,
            out_shapes=[
                C.networks.latent_shape,
            ],
            **kwargs,
        )

    def forward(
        self,
        *observations: Tensor,
    ) -> Tensor:
        return self.fc_split(*observations)[0]


class FcPrediction(PredictionNet):
    def __init__(self, num_blocks: int, block_depth: int = 2, **kwargs: Any):
        super().__init__()
        from config import C

        input_width = sum(
            map(
                math.prod,
                [
                    C.networks.latent_shape,
                    C.networks.belief_shape,
                ],
            )
        )
        width = 1 * input_width
        self.up = nn.Linear(input_width, width)
        output_shapes = [
            [C.networks.scalar_support_size],
            [C.game.instance.max_num_actions],
        ]
        self.resnet = nn.Sequential(
            *[
                ResidualBlock([BasicBlock(width) for _ in range(block_depth)])
                for _ in range(num_blocks)
            ]
        )
        self.head = FcSplitReshapeHead(width, output_shapes)

    def forward(self, latent: Tensor, belief: Tensor) -> tuple[Tensor, Tensor]:
        x = torch.cat([latent.flatten(1), belief.flatten(1)], dim=1)
        x = self.up(x)
        x = self.resnet(x)
        return cast(tuple[Tensor, Tensor], self.head(x))


class FcDynamics(DynamicsNet):
    def __init__(
        self,
        num_shared_blocks: int,
        num_pred_blocks: int,
        num_main_blocks: int,
        block_depth: int = 2,
        **kwargs: Any,
    ):
        super().__init__()
        from mcts import TurnStatus
        from config import C

        def make_resnet(num_blocks: int) -> nn.Sequential:
            return nn.Sequential(
                *[
                    ResidualBlock([BasicBlock(width) for _ in range(block_depth)])
                    for _ in range(num_blocks)
                ]
            )

        input_width = sum(
            map(
                math.prod,
                [
                    C.networks.latent_shape,
                    C.networks.belief_shape,
                    [C.game.instance.max_num_actions],
                ],
            )
        )
        width = 1 * input_width
        self.up = nn.Linear(input_width, width)
        self.resnet_shared = make_resnet(num_shared_blocks)
        self.resnet_main = make_resnet(num_main_blocks)
        self.resnet_pred = make_resnet(num_pred_blocks)

        self.pred_head = FcSplitReshapeHead(
            width,
            [
                [C.networks.scalar_support_size],
                [C.game.instance.max_num_players + len(TurnStatus)],
            ],
        )
        self.latent_head = FcSplitReshapeHead(
            width,
            [
                C.networks.latent_shape,
                C.networks.belief_shape,
            ],
        )

    def forward(
        self, latent: Tensor, belief: Tensor, action_onehot: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        x = torch.cat([latent.flatten(1), belief.flatten(1), action_onehot], dim=1)
        x = self.up(x)
        x = self.resnet_shared(x)
        a = self.resnet_main(x)
        b = self.resnet_pred(x)
        return cast(
            tuple[Tensor, Tensor, Tensor, Tensor],
            tuple(self.latent_head(a) + self.pred_head(b)),
        )


"""
class FcDynamics(DynamicsNet):
    def __init__(self, **kwargs: Any):
        super().__init__()
        from mcts import TurnStatus
        from config import C

        self.fc_split = FcSplitOutputs(
            in_shapes=[
                C.networks.latent_shape,
                C.networks.belief_shape,
                [C.game.instance.max_num_actions],
            ],
            out_shapes=[
                C.networks.latent_shape,
                C.networks.belief_shape,
                [C.networks.scalar_support_size],
                [C.game.instance.max_num_players + len(TurnStatus)],
            ],
            **kwargs,
        )

    def forward(
        self,
        latent: Tensor,
        belief: Tensor,
        action_onehot: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return cast(
            tuple[Tensor, Tensor, Tensor, Tensor],
            self.fc_split(latent, belief, action_onehot),
        )

"""

# ========================================

"""

class FcDynamics2(DynamicsNet):
    def __init__(
        self,
        main_residual_blocks: int,
        shared_residual_blocks: int,
        pred_residual_blocks: int = 2,
        #side_widths: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__()
        from mcts import TurnStatus
        from config import C

        pred_shapes = [
            [C.networks.scalar_support_size],
            [C.game.instance.max_num_players + len(TurnStatus)],
        ]
        '''
        if side_widths is None:
            side_widths = (
                2 * sum(map(math.prod, pred_shapes)) // min(main_residual_blocks, 2)
            )
        pred_width = main_residual_blocks * side_widths
        '''
        signal_width = sum(
            map(
                math.prod,
                [
                    C.networks.latent_shape,
                    C.networks.belief_shape,
                ],
            )
        )
        self.shared_blocks = nn.ModuleList(
            [XResidualBlock([BasicBlock(signal_width+C.game.instance.max_num_actions, signal_width) for _ in range(2)])
            for _ in range(shared_residual_blocks)]
        )
        self.repr_blocks = nn.Sequential(
            *[FcResidualXBlock(signal_width) for _ in range(main_residual_blocks)]
        )

        self.pred_blocks = nn.Sequential(
            *[FcResidualXBlock(signal_width) for _ in range(pred_residual_blocks)]
        )
        self.pred_head = FcSplitReshapeHead(signal_width, pred_shapes)
        self.splitter = SplitReshape(
            [
                C.networks.latent_shape,
                C.networks.belief_shape,
            ],
        )

    def forward(
        self,
        latent: Tensor,
        belief: Tensor,
        action_onehot: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        x = torch.cat([latent.flatten(1), belief.flatten(1)], dim=1)
        for block in self.shared_blocks:
            x = block(x, action_onehot)
        y = self.repr_blocks(x)
        z = self.pred_blocks(x)
        # y = torch.cat(side_outputs, dim=1)
        # y=self.pred_blocks(y)
        return cast(
            tuple[Tensor, Tensor, Tensor, Tensor],
            tuple(self.splitter(y) + self.pred_head(z)),
        )
"""
