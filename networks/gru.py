from typing import Any
from collections.abc import Mapping

from torch import Tensor, nn

from .bases import DynamicsNet
from .fully_connected import FcReshaper


class GRUDynamics(DynamicsNet):
    def __init__(self, fc_head_args: Mapping[str, Any] = dict()):
        from mcts import TurnStatus
        from config import C

        super().__init__()
        assert len(C.networks.latent_shape) == 1
        latent_size = C.networks.latent_shape[0]

        self.gru = nn.GRU(
            input_size=C.game.instance.max_num_actions,
            hidden_size=latent_size,
            num_layers=1,
        )

        self.fc_reshape = FcReshaper(
            in_shapes=[
                [C.game.instance.max_num_actions],
                [latent_size],
            ],
            out_shapes=[
                [C.networks.scalar_support_size, C.game.instance.max_num_players],
                [C.game.instance.max_num_players + len(TurnStatus)],
            ],
            **fc_head_args,
        )

    def forward(
        self,
        latent_in: Tensor,
        action_onehot: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        def gru_reshape(
            gru: nn.Module, input: Tensor, h0: Tensor
        ) -> tuple[Tensor, Tensor]:
            out, hn = gru(input.unsqueeze(0), h0.unsqueeze(0).contiguous())
            return out.squeeze(0), hn.squeeze(0)

        reward, turn = self.fc_reshape(action_onehot, latent_in)
        _, latent_out = gru_reshape(self.gru, action_onehot, latent_in)
        return latent_out, reward, turn
