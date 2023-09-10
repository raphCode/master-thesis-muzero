from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeAlias, cast
from collections.abc import Sequence

from config import C
from rl_player import RLBase
from games.bases import Player

if TYPE_CHECKING:
    from networks import Networks

PlayerPartial: TypeAlias = functools.partial[Player]


class PCBase(ABC):
    """
    A PlayerController manages normal Players, RLPlayers and their networks.
    A single instance is alive during selfplay and performs the following functions:
    - instantiate all players from partials
    - decide which RLPlayers share their networks
    - assign players to new game matches (participants and player ids / order)
    - update RLPlayer networks (propagate learned parameters to selfplay)
    """

    def __init__(
        self,
        player_partials: Sequence[PlayerPartial],
    ):
        pass

    @abstractmethod
    def get_players(self) -> Sequence[Player]:
        """
        Sequence of players that should take part in a game, called for each match.
        """
        pass

    @abstractmethod
    def update_networks(self, new_networks: Networks) -> None:
        """
        Update network parameters of RLPlayers with new information from training.
        """
        pass


class SelfplayPC(PCBase):
    """
    Uses the same RLPlayer configuration for all players.
    Requires all matches to have the same (maximum) number of players.
    This arrangement works for single- and multi-player games.
    """

    net: Networks
    players: list[RLBase]

    def __init__(self, player_partials: Sequence[PlayerPartial]):
        assert len(player_partials) == 1
        self.net = C.networks.factory()
        self.players = [
            cast(RLBase, player_partials[0](self.net))
            for _ in range(C.game.instance.max_num_players)
        ]
        assert isinstance(self.players[0], RLBase)

    def get_players(self) -> list[RLBase]:
        return self.players

    def update_networks(self, new_networks: Networks) -> None:
        pass
