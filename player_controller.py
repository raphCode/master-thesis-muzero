import functools
from abc import ABC, abstractmethod
from typing import TypeAlias
from collections.abc import Sequence

from rl_player import RLBase
from games.bases import Player, MatchData
from networks.bases import Networks

PlayerPartial: TypeAlias = functools.partial[Player] | functools.partial[RLBase]


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
    def get_players(self, match_data: MatchData) -> Sequence[Player | RLBase]:
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
