import math
import itertools
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from utils import optional_map
from config import C
from networks.bases import (
    DynamicsNet,
    PredictionNet,
    DynamicsReturn,
    PredictionReturn,
    RepresentationNet,
    RepresentationReturn,
)


class FcBase(nn.Module):
    def __init__(
        self,
        input_width: int,
        output_width: int,
        hidden_depth: int = 2,
        width: Optional[int] = None,
    ):
        super().__init__()
        if width is None:
            width = input_width
        widths = [input_width] + [width] * hidden_depth + [output_width]
        self.fc_layers = [nn.Linear(a, b) for a, b in itertools.pairwise(widths)]
        for n, layer in enumerate(self.fc_layers):
            self.add_module(f"fc{n}", layer)

    def fc_forward(self, *inputs: Optional[Tensor]) -> Tensor:
        x = torch.cat([i.flatten(1) for i in inputs if i is not None], dim=1)
        for fc in self.fc_layers:
            x = fc(x)
            tmp = x
            x = F.relu(x)
        return tmp


class FcRepresentation(FcBase, RepresentationNet):
    def __init__(self, **kwargs: Any):
        super().__init__(
            input_width=sum(map(math.prod, C.game.instance.observation_shapes)),
            output_width=math.prod(C.networks.latent_shape),
            **kwargs,
        )

    def forward(
        self,
        *observations: Tensor,
        **kwargs: Any,
    ) -> RepresentationReturn:
        return self.fc_forward(*observations)


class FcPrediction(FcBase, PredictionNet):
    def __init__(self, **kwargs: Any):
        input_sizes = (
            optional_map(math.prod)(C.networks.belief_shape) or 0,
            math.prod(C.networks.latent_shape),
        )
        self.output_sizes = [
            1,
            C.game.instance.max_num_actions,
            C.game.instance.max_num_players + C.game.instance.has_chance_player,
        ]
        super().__init__(
            input_width=sum(input_sizes),
            output_width=sum(self.output_sizes),
            **kwargs,
        )

    def forward(
        self, latent: Tensor, belief: Optional[Tensor], logits: bool = False
    ) -> PredictionReturn:
        result = self.fc_forward(latent, belief)
        value, policy, player_type = torch.split(result, self.output_sizes, dim=1)
        if logits:
            return value, policy, player_type
        return value, F.softmax(policy, dim=1), F.softmax(player_type, dim=1)


class FcDynamics(FcBase, DynamicsNet):
    def __init__(self, **kwargs: Any):
        belief_size = optional_map(math.prod)(C.networks.belief_shape) or 0
        input_sizes = (
            math.prod(C.networks.latent_shape),
            belief_size,
            C.game.instance.max_num_actions,
        )
        self.output_sizes = [
            math.prod(C.networks.latent_shape),
            belief_size,
            1,
        ]
        super().__init__(
            input_width=sum(input_sizes),
            output_width=sum(self.output_sizes),
            **kwargs,
        )

    def forward(
        self, latent: Tensor, belief: Optional[Tensor], action_onehot: Tensor
    ) -> DynamicsReturn:
        result = self.fc_forward(latent, belief, action_onehot)
        latent, belief, reward = torch.split(result, self.output_sizes, dim=1)
        return latent, belief if belief.numel() > 0 else None, reward
