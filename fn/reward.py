from typing import TypeAlias
from collections.abc import Callable

from games.bases import GameState

RewardFn: TypeAlias = Callable[[GameState, int], float]


def assert_fn_type(fn: RewardFn) -> None:
    pass


def default(state: GameState, player_id: int) -> float:
    return state.rewards[player_id]
