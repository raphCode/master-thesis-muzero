import math
import itertools
from typing import Any
from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from util import get_output_shape

from .util import ModuleFactory
from .bases import RepresentationNet
from .fully_connected import GenericFc


class ConvRepresentation(RepresentationNet):
    def __init__(
        self,
        filters: int,
        depth: int,
        latent_features: int,
        fc_head_args: Mapping[str, Any] = dict(),
        kernel_size: int = 3,
        padding: str = "same",
        **kwargs: Any,
    ):
        super().__init__()

        from config import C

        image_shape, *other_obs_shapes = C.game.instance.observation_shapes
        assert len(image_shape) == 3, "First observation must be 3D CxHxW image Tensor"
        assert depth >= 1
        in_channels = image_shape[0]
        # channels = [in_channels] + depth * [filters]
        conv_factory = ModuleFactory(self, nn.Conv2d, "conv")
        # norm_factory = ModuleFactory(self, nn.BatchNorm2d, "bn")
        self.upconv = nn.Conv2d(in_channels, filters, kernel_size=1)
        self.convs = [
            conv_factory(a, b, kernel_size, padding=padding)
            for a, b in itertools.pairwise([filters] * depth)
        ]
        # self.norms = [norm_factory(c) for c in channels[1:]]
        self.softmax_w = nn.Softmax(dim=-1)
        self.softmax_h = nn.Softmax(dim=-2)

        conv_out_shape = get_output_shape(self.conv_forward, image_shape)
        self.fc_head = GenericFc(
            input_width=sum(map(math.prod, [conv_out_shape] + other_obs_shapes)),
            output_width=latent_features,
            first_layer_pre_activation=False,
            **fc_head_args,
        )

    def conv_forward(self, x: Tensor) -> Tensor:
        features: list = []

        def global_maxpool(x: Tensor) -> Tensor:
            return x.max(dim=-1).values.max(dim=-1).values

        x = self.upconv(x)
        first = self.convs[0]
        # for conv, norm in zip(self.convs, self.norms):
        for conv in self.convs:
            skip = x
            # if conv is not first:
            # x = norm(x)
            x = F.relu(x)
            x = conv(x)
            # features.append(global_maxpool(x))
            if skip.shape == x.shape:
                x = x + skip
        # return torch.cat([self.softmax_w(x), self.softmax_h(x)], dim=-1)
        return torch.cat([x.max(dim=-1).values, x.max(dim=-2).values], dim=-1)
        return x.max(dim=-1).values.max(dim=-1).values
        return torch.cat(features, dim=-1)

    def forward(
        self,
        *observations: Tensor,
    ) -> Tensor:
        image_obs, *more_obs = observations
        conv_out = self.conv_forward(image_obs)
        return self.fc_head(conv_out, *more_obs)
