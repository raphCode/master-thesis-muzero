from typing import TypeAlias
from collections.abc import Callable

RewardFn: TypeAlias = Callable[[tuple[float, ...], int], float]


def assert_fn_type(fn: RewardFn) -> None:
    pass


def default(rewards: tuple[float, ...], player_id: int) -> float:
    return rewards[player_id]
