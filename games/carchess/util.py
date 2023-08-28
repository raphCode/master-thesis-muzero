from __future__ import annotations

from enum import IntEnum
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

Pos: TypeAlias = tuple[int, int]
ndarr_bool: TypeAlias = npt.NDArray[np.bool_]
ndarr_int: TypeAlias = npt.NDArray[np.int64]
ndarr_f64: TypeAlias = npt.NDArray[np.float64]


class Dir(IntEnum):
    North = 0
    East = 1
    South = 2
    West = 3

    def turn(self, offset: int) -> Dir:
        return Dir((self + offset) % len(Dir))

    @property
    def counter_clockwise(self) -> Dir:
        return self.turn(-1)

    @property
    def clockwise(self) -> Dir:
        return self.turn(1)

    def offset(self, pos: Pos) -> Pos:
        offset = {
            Dir.North: (-1, 0),
            Dir.East: (0, 1),
            Dir.South: (1, 0),
            Dir.West: (0, -1),
        }[self]
        return tuple(np.array(pos) + offset)  # type: ignore [return-value]
