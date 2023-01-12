from typing import TypeAlias
from collections.abc import Callable

reward_fn: TypeAlias = Callable[[tuple[float, ...], int], float]


def default(rewards: tuple[float, ...], player_id: int) -> float:
    return rewards[player_id]
