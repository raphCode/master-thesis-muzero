import math
import itertools
from typing import Any, Optional, cast
from collections.abc import Sequence, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .bases import NetBase, DynamicsNet, PredictionNet, RepresentationNet
from functools import partial
from .util import ModuleFactory
from util import get_output_shape
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
        channels = [in_channels] + depth * [filters]
        conv_factory = ModuleFactory(self, nn.Conv2d, "conv")
        norm_factory = ModuleFactory(self, nn.BatchNorm2d, "bn")
        pool_factory = ModuleFactory(self, nn.MaxPool2d, "pool")
        self.convs = [
            conv_factory(a, b, kernel_size, padding=padding)
            for a, b in itertools.pairwise(channels)
        ]
        self.norms = [norm_factory(c) for c in channels[1:]]
        self.pools = [pool_factory(2, ceil_mode=True) for _ in range(depth)]

        conv_out_shape = get_output_shape(self.conv_forward, image_shape)
        self.fc_head = GenericFc(
            input_width=sum(map(math.prod, [conv_out_shape] + other_obs_shapes)),
            output_width=latent_features,
            raw_out=False,
            **fc_head_args,
        )

    def conv_forward(self, x: Tensor) -> Tensor:
        features :list=[]
        def global_maxpool(x:Tensor)->Tensor:
            return x.max(dim=-1).values.max(dim=-1).values

        first = self.convs[0]
        for conv, norm in zip(self.convs, self.norms):
            skip = x
            if conv is not first:
                x = norm(x)
                x = F.relu(x)
            x = conv(x)
            features.append(global_maxpool(x))
            if skip.shape == x.shape:
                x = x + skip
        return x.max(dim=-1).values.max(dim=-1).values
        return torch.cat(features, dim=-1)
        return torch.cat([x.max(dim=-1).values, x.max(dim=-2).values], dim=-1)

    def forward(
        self,
        *observations: Tensor,
    ) -> Tensor:
        image_obs, *more_obs = observations
        conv_out = self.conv_forward(image_obs)
        return self.fc_head(conv_out, *more_obs)
