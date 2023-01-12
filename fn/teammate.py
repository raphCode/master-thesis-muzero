from typing import TypeAlias
from collections.abc import Callable

teammate_fn: TypeAlias = Callable[[int, int], bool]


def no(pid_a: int, pid_b: int) -> bool:
    return False
