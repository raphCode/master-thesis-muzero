import math
import itertools
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from config import C
from trajectory import PlayerType
from networks.bases import DynamicsNet, PredictionNet, RepresentationNet


class ConvBase(ABC):
    def __init__(
        self,
        input_shape: tuple[int],
        kernel_size: int,
        channels: int,
        depth: int,
        fc_head: bool,
        fc_output_width: Optional[int] = None,
        fc_hidden_width: Optional[int] = None,
        fc_depth: int = 0,
        residual_skip_length: int = 2,
        **convolution_args,
    ):
        super().__init__()

        self.residual_skip_length = residual_skip_length
        assert (
            len(input_shape) == 3
        ), "convolutional networks need inputs of shape (C, H, W)"

        in_channels = [input_shape[0]] + [channels] * (depth - 1)
        self.conv_layers = [
            nn.Conv2d(
                in_features=in_ch,
                out_features=channels,
                kernel_size=kernel_size,
                padding="same",
            )
            for in_ch in in_channels
        ]
        for n, layer in enumerate(self.conv_layers):
            self.add_module(f"conv{n}", layer)

        self.fc_head = fc_head
        if fc_head:
            assert (
                fc_output_width is not None
            ), "for the fully connected output fc_output_width must be specified"
            assert (
                fc_depth == 0 or fc_hidden_width is not None
            ), "for fc_depth != 0 the hidden width must be specified"

        widths = (
            [input_shape[1] * input_shape[2] * channels]
            + [fc_hidden_width] * fc_depth
            + [fc_output_width]
        )
        self.fc_layers = [nn.Linear(a, b) for a, b in itertools.pairwise(widths)]
        for n, layer in enumerate(self.fc_layers):
            self.add_module(f"fc{n}", layer)

    @abstractmethod
    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        x = torch.cat([i.flatten(1) for i in inputs], dim=1)
        skip_x = None
        for n, conv in self.conv_layers:
            x = conv(x)
            if (n - 1) % self.residual_skip_length == 0:
                if skip_x is not None:
                    x += skip_x
                skip_x = x

            tmp = x
            x = F.relu(x)

        if self.output_width is None:
            return tmp

        x = x.flatten()
        for fc in self.fc_layers:
            x = fc(x)
            tmp = x
            x = F.relu(x)
        return tmp


class ConvRepresentation(ConvBase, RepresentationNet):
    def __init__(self, *args, **kwargs):
        self.output_sizes = (
            C.nets.latent_rep_shape[0],
            C.nets.beliefs_shape[0],
        )
        super().__init__(
            *args,
            input_shape=C.game.instance.observation_shapes[0],
            **kwargs,
        )

    def forward(
        self,
        obs_image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = super().forward(obs_image)
        return torch.split(out, self.output_sizes, dim=1)


class ConvPrediction(ConvBase, PredictionNet):
    def __init__(self, *args, **kwargs):
        assert (
            C.nets.latent_rep_shape[1:] == C.nets.beliefs_shape[1:]
        ), "H and W dimensions of latent representation and beliefs must match"
        input_shape = [
            C.nets.latent_rep_shape[0] + C.nets.beliefs_shape[0]
        ] + C.nets.latent_rep_shape[1:]
        self.output_sizes = (
            1,
            C.game.instance.max_num_actions,
            len(PlayerType),
        )
        super().__init__(
            *args,
            input_shape=input_shape,
            fc_head=True,
            fc_output_width=sum(self.output_sizes),
            **kwargs,
        )

    def forward(
        self, latent_rep: torch.Tensor, beliefs: torch.Tensor, logits=False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = super().forward(torch.cat((latent_rep, beliefs), dim=1))
        value, policy, player_type = torch.split(out, self.output_sizes, dim=1)
        if logits:
            return value, policy, player_type
        return value, F.softmax(policy, dim=1), F.softmax(player_type, dim=1)


class ConvDynamics(ConvBase, DynamicsNet):
    def __init__(self, *args, **kwargs):
        obs_shape = C.game.instance.observation_shapes[0]
        assert (
            C.game.instance.max_num_actions % (obs_shape[1] * obs_shape[2]) == 0
        ), "Cannot reshape the action onehot into the observation image size (H*W)"
        assert (
            C.nets.latent_rep_shape[1:] == C.nets.beliefs_shape[1:]
        ), "H and W dimensions of latent representation and beliefs must match"

        input_shape = [
            C.nets.latent_rep_shape[0]
            + C.nets.beliefs_shape[0]
            + C.game.instance.max_num_actions / (obs_shape[1] * obs_shape[2])
        ] + C.nets.latent_rep_shape[1:]
        self.output_sizes = (
            C.nets.latent_rep_shape[0],
            C.nets.beliefs_shape[0],
        )
        super().__init__(
            *args,
            input_shape=input_shape,
            fc_head=False,
            **kwargs,
        )
        self.fc_reward=nn.Linear(

    def forward(
        self, latent_rep: torch.Tensor, beliefs: torch.Tensor, action_onehot: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_shape = C.game.instance.observation_shapes[0]
        action_image= action_onehot.reshape((obs_shape[1] * obs_shape[2]))

        out = super().forward(torch.cat((latent_rep, beliefs, action_image), dim=1))
        latent_rep, beliefs, action_image = torch.split(out, self.output_sizes, dim=1)
        return latent_rep, beliefs, action_image.flatten()
