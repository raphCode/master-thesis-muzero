from __future__ import annotations

from copy import copy
from typing import Any, cast
from functools import cached_property

import numpy as np
import torch

from .openspiel_wrapper import OpenSpielGame, OpenSpielGameState


class TwentyEightyFourGameState(OpenSpielGameState):
    game: TwentyEightyFourGame

    @property
    def observation(self) -> tuple[torch.Tensor]:  # type: ignore [override]
        shape = self.game.observation_shapes[0]
        (n, *size) = shape
        board = np.array(self.state.observation_tensor()).reshape(size)
        assert np.all(board <= 2**n)
        ret = np.full(shape, -1, dtype=np.float32)
        for x in range(n):
            mask = board == 2 ** (x + 1)
            ret[x][mask] = 1
        return (torch.tensor(ret),)

    @property
    def _external_legal_actions(self) -> set[int]:
        return set(self.state.legal_actions())

    @property
    def _internal_legal_actions(self) -> set[int]:
        ret = set(self.state.legal_actions())
        if self.game.abort_noops and self.state.is_player_node():
            for a in copy(ret):
                if self.state.child(a).is_player_node():
                    # a successfull movement spawns a new tile => chance node
                    ret.remove(a)
        return ret


class TwentyEightyFourGame(OpenSpielGame):
    abort_noops: int
    layers: int

    def __init__(
        self,
        abort_noops: Any = True,
        layers: Any = 11,
        **kwargs: dict[str, Any],
    ):
        self.abort_noops = bool(abort_noops)
        self.layers = int(layers)
        super().__init__(game_name="2048", **kwargs)

    def new_initial_state(self) -> TwentyEightyFourGameState:
        return TwentyEightyFourGameState(
            self.game.new_initial_state(),
            game=self,
        )

    @cached_property
    def observation_shapes(self) -> tuple[tuple[int, int, int]]:  # type: ignore [override] # noqa: E501
        shape = cast(tuple[int, int], tuple(self.game.observation_tensor_shape()))
        assert len(shape) == 2
        return ((self.layers, *shape),)
