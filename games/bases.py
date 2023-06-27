from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypedDict
from functools import lru_cache
from collections import defaultdict
from collections.abc import Sequence, Collection

import torch
from attrs import frozen

if TYPE_CHECKING:
    from util import ndarr_f64, ndarr_bool


class Teams:
    """
    Contains information about which players are in a team.
    """

    teams: defaultdict[int, set[int]]

    def __init__(self, team_definitions: Collection[Collection[int]]):
        assert all(
            len(t) > 1 for t in team_definitions
        ), "All teams must have at least 2 players"
        for ta, tb in itertools.combinations(team_definitions, 2):
            assert set(ta).isdisjoint(tb), f"Teams are not disjoint: {ta} and {tb}"
        self.teams = defaultdict(set)
        for team in team_definitions:
            for a, b in itertools.permutations(set(team), 2):
                self.teams[a].add(b)

    @lru_cache(maxsize=1024)
    def __contains__(self, item: object) -> bool:
        """
        Tests wheter the given sequence of player ids are in the same team.
        """
        assert isinstance(item, Sequence)
        assert len(item) > 1
        assert all(isinstance(x, int) for x in item)
        member, *rest = item
        return self.teams[member].issuperset(rest)

    def __getitem__(self, item: object) -> frozenset[int]:
        """
        Return the other team members for this player id.
        """
        assert isinstance(item, int)
        return frozenset(self.teams[item])


@frozen
class MatchData:
    """
    Data which is static for a single match / playout.
    """

    num_players: int
    teams: Teams


class GameStateInitKwArgs(TypedDict):
    game: Game
    match_data: MatchData


class GameState(ABC):
    """
    A state in an active game / match. It keeps all the information necessary for all
    players and is mutated on moves.
    """

    match_data: MatchData
    game: Game

    def __init__(self, game: Game, match_data: MatchData):
        self.game = game
        self.match_data = match_data

    @property
    @abstractmethod
    def observation(self) -> tuple[torch.Tensor, ...]:
        """
        Return observation tensors. Multiple tensors are possible for different shapes.
        """
        pass

    @property
    @abstractmethod
    def valid_actions_mask(self) -> ndarr_bool:
        """
        Specifies wich actions are valid to take in the current state.
        Length of returned array should be game.max_num_actions
        """
        pass

    @property
    @abstractmethod
    def rewards(self) -> tuple[float, ...]:
        """
        Tuple of rewards, one for each player, starting at 0 for the first player
        """
        pass

    @property
    @abstractmethod
    def is_terminal(self) -> bool:
        pass

    @property
    @abstractmethod
    def is_chance(self) -> bool:
        pass

    @property
    @abstractmethod
    def current_player_id(self) -> int:
        """
        Return the zero-based index of the player currently at turn.
        This may raise an exception when called on chance or terminal nodes.
        """
        pass

    @property
    @abstractmethod
    def chance_outcomes(self) -> ndarr_f64:
        """
        On chance events, this can be called to get the action probabilities.
        Length of returned array should be game.max_num_actions
        """
        pass

    @abstractmethod
    def apply_action(self, action: int) -> None:
        """
        Mutate the game state and advance the game by making the given move.
        """
        pass


class Game(ABC):
    """
    Factory for game matches / playouts. Holds information valid across different matches.
    Matches may differ in player count or team groups, allowing the network to generalize
    across different player configurations.
    """

    @abstractmethod
    def new_initial_state(self) -> GameState:
        """
        Return a new GameState which represents a game / match at intial state
        """
        pass

    @property
    @abstractmethod
    def observation_shapes(self) -> tuple[tuple[int, ...], ...]:
        """
        Shape of observation tensors. Outer tuple is for defining multiple tensors.
        Used to set the size of the prediction network inputs.
        """
        pass

    @property
    @abstractmethod
    def max_num_actions(self) -> int:
        """
        Number of actions or chance outcomes, whichever is higher.
        Used to set the size of the dynamics network inputs.
        """
        pass

    @property
    @abstractmethod
    def max_num_players(self) -> int:
        """
        Maximum number of players in a match, excluding the chance player.
        Used to set the size of reward tuples, the team matrix and dynamics network
        outputs.
        """
        pass


class Player(ABC):
    @abstractmethod
    def request_action(self, state: GameState) -> int:
        """Request an action from the player for the current game state"""
        pass
