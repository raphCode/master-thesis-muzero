from __future__ import annotations

from copy import copy
from typing import Any

from .openspiel_wrapper import OpenSpielGame, OpenSpielGameState


class TwentyEightyFourGameState(OpenSpielGameState):
    game: TwentyEightyFourGame

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

    def __init__(
        self,
        abort_noops: Any = True,
        **kwargs: dict[str, Any],
    ):
        self.abort_noops = bool(abort_noops)
        super().__init__(game_name="2048", **kwargs)

    def new_initial_state(self) -> TwentyEightyFourGameState:
        return TwentyEightyFourGameState(
            self.game.new_initial_state(),
            game=self,
        )
