import itertools
from typing import Any, Optional, cast
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from util import broadcast_cat, copy_type_signature
from networks.bases import DynamicsNet, PredictionNet, RepresentationNet

from .fully_connected import (
    BasicBlock,
    ResidualBlock,
    FcSplitOutputs,
    raph_relu,
    skip_connections,
)


def get_output_shape(module: nn.Module, input_shape: Sequence[int]) -> Sequence[int]:
    example_tensor = torch.zeros(2, *input_shape)
    return cast(Sequence[int], module(example_tensor).shape[1:])


class GenericConv(nn.Module):
    """
    Generic 2d convolution network implementation.
    """

    out_shape: tuple[int, ...]

    def __init__(
        self,
        input_shape: Sequence[int],
        *,
        hidden_depth: int = 0,
        kernel_size: int = 3,
        num_channels: Optional[int] = None,
        output_channels: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__()
        assert len(input_shape) == 3
        if num_channels is None:
            num_channels = 2 * input_shape[0]
        if output_channels is None:
            output_channels = num_channels
        channels = [input_shape[0]] + [num_channels] * hidden_depth + [output_channels]
        self.conv_layers = [
            nn.Conv2d(a, b, kernel_size, **kwargs)
            for a, b in itertools.pairwise(channels)
        ]
        for n, layer in enumerate(self.conv_layers):
            self.add_module(f"conv{n}", layer)
        self.out_shape = tuple(get_output_shape(self, input_shape))

    def forward(self, x: Tensor) -> Tensor:
        return skip_connections(self.conv_layers, x)

    # method overrides are to provide properly typed function signatures:
    @copy_type_signature(forward)
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return super().__call__(*args, **kwargs)


def get_conv_shapes() -> tuple[list[int], list[int]]:
    """
    Checks shapes of belief and latent, returns the concatenated shape and list of ints to split them again.
    """
    from config import C

    latent_shape = C.networks.latent_shape
    belief_shape = C.networks.belief_shape
    assert len(latent_shape) == 3
    belief_channels = 0
    if not belief_shape == (0,):
        assert len(belief_shape) == 3
        assert latent_shape[1:] == belief_shape[1:]
        belief_channels = belief_shape[0]
    combined_shape = [
        latent_shape[0] + belief_shape[0],
    ] + list(
        latent_shape
    )[1:]
    return combined_shape, [latent_shape[0], belief_channels]


class ConvRepresentation(RepresentationNet):
    pool_after: set[int]

    def __init__(
        self,
        channels: int,
        layers: int,
        pool_after: list[int] = [],
        kernel_size: int = 3,
        padding: str = "same",
        encode_position: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        from config import C

        self.encode_position = bool(encode_position)
        observation_shapes = C.game.instance.observation_shapes
        assert len(observation_shapes) == 1
        assert len(observation_shapes[0]) == 3, "Game does not provide a 3D observation"
        obs_channels = observation_shapes[0][0] + 2 * self.encode_position
        self.upconv = nn.Conv2d(obs_channels, channels, 1)
        self.conv_layers = nn.ModuleList(
            [
                nn.Conv2d(
                    # obs_channels if n == 0 else channels,
                    channels,
                    channels,
                    kernel_size,
                    padding=padding,
                    **kwargs,
                )
                for n in range(layers)
            ]
        )
        self.input_bn = nn.BatchNorm2d(observation_shapes[0][0])
        self.batchnorms = nn.ModuleList([nn.BatchNorm2d(channels) for _ in range(layers)])
        self.downconv = nn.Conv2d(channels, C.networks.latent_shape[0], 1)
        self.pool_after = set(pool_after)
        self.bn_out = nn.BatchNorm1d(C.networks.latent_shape[0])
        width = C.networks.latent_shape[0] * 10
        self.fc = nn.Sequential(
            *[ResidualBlock([BasicBlock(width) for _ in range(2)]) for _ in range(5)]
        )
        # self.pool = nn.MaxPool2d(2, ceil_mode=True)
        # TODO conv out shape may not be the latent directly
        # TODO: better error messages
        out_shape = get_output_shape(self, observation_shapes[0])
        assert (
            out_shape == C.networks.latent_shape
        ), f"shape mismatch:\nrepr net output shape: {out_shape}\nlatent shape: {C.networks.latent_shape}"

    def forward(
        self,
        *observations: Tensor,
    ) -> Tensor:
        x = observations[0]
        x = self.input_bn(x)
        if self.encode_position:
            n = x.shape[0]
            w, h = x.shape[-2:]
            pos_w = torch.linspace(-1, 1, w).view(1, 1, -1, 1).expand(n, 1, w, h)
            pos_h = torch.linspace(-1, 1, h).view(1, 1, 1, -1).expand(n, 1, w, h)
            x = torch.cat([x, pos_w, pos_h], dim=1)
        x = self.upconv(x)
        for n, (conv, bn) in enumerate(zip(self.conv_layers, self.batchnorms)):
            skip = x
            # x = F.celu(x, alpha=0.1)  # pre-activation
            x = raph_relu(x)
            x = bn(x)
            x = conv(x)
            if skip.shape == x.shape:
                x = x + skip
            if n + 1 in self.pool_after:
                x = F.max_pool2d(x, 2, ceil_mode=True)
        # x = x.mean(dim=[-1, -2])
        # Either fc output or down convolution
        # x = self.downconv(x)
        # x = self.fc(x)
        # x = F.celu(x, alpha=0.1)
        a = x.max(dim=-1).values
        b = x.max(dim=-2).values
        x = self.bn_out(torch.cat([a, b], dim=-1))
        return self.fc(x.flatten(1)).reshape([-1, 16, 10])


class ConvPrediction(PredictionNet):
    conv: GenericConv
    fc: FcSplitOutputs
    split_sizes: list[int]

    def __init__(
        self,
        fully_connected_head_args: dict[str, Any] = {},
        **kwargs: Any,
    ):
        super().__init__()
        from config import C

        combined_shape, self.split_sizes = get_conv_shapes()
        self.conv = GenericConv(input_shape=combined_shape, **kwargs)
        self.fc = FcSplitOutputs(
            in_shapes=[self.conv.out_shape],
            out_shapes=[
                [1],
                [C.game.instance.max_num_actions],
            ],
            **fully_connected_head_args,
        )

    def forward(self, latent: Tensor, belief: Tensor) -> tuple[Tensor, Tensor]:
        conv_out = self.conv(broadcast_cat(latent, belief, dim=1))
        return cast(
            tuple[Tensor, Tensor],
            self.fc(conv_out),
        )


class ConvDynamics(DynamicsNet):
    conv: GenericConv
    fc: FcSplitOutputs
    split_sizes: list[int]

    def __init__(
        self,
        fully_connected_head_args: dict[str, Any] = {},
        **kwargs: Any,
    ):
        super().__init__()
        from mcts import TurnStatus
        from config import C

        combined_shape, self.split_sizes = get_conv_shapes()
        combined_shape[0] += C.game.instance.max_num_actions
        self.conv = GenericConv(
            input_shape=combined_shape,
            output_channels=C.networks.latent_shape[0],
            **kwargs,
        )
        self.fc = FcSplitOutputs(
            in_shapes=[self.conv.out_shape],
            out_shapes=[
                [1],
                [C.game.instance.max_num_players + len(TurnStatus)],
            ],
            **fully_connected_head_args,
        )

    def forward(
        self,
        latent: Tensor,
        belief: Tensor,
        action_onehot: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        conv_out = self.conv(broadcast_cat(latent, belief, action_onehot, dim=1))
        fc_out = self.fc(conv_out)
        return cast(
            tuple[Tensor, Tensor, Tensor, Tensor],
            tuple(torch.split(conv_out, self.split_sizes, dim=1)) + fc_out,
        )
