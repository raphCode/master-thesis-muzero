from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeAlias
from collections.abc import Sequence

from config import C
from rl_player import RLBase
from games.bases import Player, MatchData

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
    def get_players(self, match_data: MatchData) -> Sequence[Player]:
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


class SinglePC(PCBase):
    """
    Simple Player Controller for one RLPlayer in single-player games.
    """

    net: Networks
    player: RLBase

    def __init__(self, player_partials: Sequence[PlayerPartial]):
        assert len(player_partials) == 1
        assert C.game.instance.max_num_players == 1
        self.net = C.networks.factory()
        player = player_partials[0](self.net)
        assert isinstance(player, RLBase)
        self.player = player

    def get_players(self, _: MatchData) -> list[RLBase]:
        return [self.player]

    def update_networks(self, new_networks: Networks) -> None:
        pass
