from os import path
from enum import IntEnum
from typing import *
from functools import cache

import numpy as np
import imageio.v3 as iio

map_path = "maps/map1"

Pos: TypeAlias = tuple[int, int]

rng = np.random.default_rng()


class Dir(IntEnum):
    North = 0
    East = 1
    South = 2
    West = 3

    def turn(self, offset: int) -> Self:
        return Dir((self + offset) % len(Dir))

    @property
    def counter_clockwise(self) -> Self:
        return self.turn(-1)

    @property
    def clockwise(self) -> Self:
        return self.turn(1)

    def offset(self, pos: Pos) -> Pos:
        pos = np.array(pos)
        offset = {
            Dir.North: (-1, 0),
            Dir.East: (0, 1),
            Dir.South: (1, 0),
            Dir.West: (0, -1),
        }[self]
        return tuple(pos + offset)


class Layer:
    size: tuple[int, int]
    pos: list[Pos]
    cars: set[int]

    def __init__(self, mask, prepopulate: float = 0.2):
        def is_at_border(x, y) -> bool:
            mx, my = mask.shape
            return x in (0, mx - 1) or y in (0, my - 1)

        def highspot_to_start_info(x, y) -> tuple[Pos, Dir]:
            mx, my = mask.shape
            if x == 0:
                # top border
                return (x, y + 1), Dir.South
            if y == 0:
                # left border
                return (x - 1, y), Dir.East
            if x == mx - 1:
                # bottom border
                return (x, y - 1), Dir.North
            if y == my - 1:
                # right border
                return (x + 1, y), Dir.West
            assert False, f"Invalid highest spot: ({x}, {y})"

        self.size = tuple(mask.shape)
        highspot_pos = np.unravel_index(np.argmax(mask), mask.shape)
        pos, dir = highspot_to_start_info(*highspot_pos)

        self.pos = []
        mask[highspot_pos] = 0
        while True:
            mask[pos] = 0
            self.pos.append(pos)
            if (mask == 0).all():
                break
            # Right-handed wall follower maze solving algorithm
            for d in (dir.clockwise, dir, dir.counter_clockwise):
                next_pos = d.offset(pos)
                if mask[next_pos] > 0:
                    pos = next_pos
                    dir = d
                    break
            else:
                assert (
                    False
                ), f"Did not find next field! pos: ({pos[0]}, {pos[1]}), dir: {dir.name}"

        n = len(self.pos)
        self.cars = list((np.arange(n)[rng.random(n) < prepopulate]))
        print(self.car_map)

    @cache
    def offset_pos(self, pos, offset: int = 1) -> Pos:
        return self.pos[self.index_at(pos) + offset]

    @cache
    def index_at(self, pos) -> int:
        return self.pos.index(pos)

    @cache
    def dir_at(self, pos) -> Dir:
        return Dir(pos - self.offset_pos(pos, 1))

    def is_car(self, pos) -> bool:
        return self.index_at(pos) in self.cars

    @property
    def lane_map(self):
        ret = np.zeros(self.size)
        for pos in self.pos:
            ret[pos] = 1
        return ret

    @property
    def car_map(self):
        ret = np.zeros(self.size)
        for index in self.cars:
            ret[self.pos[index]] = 1
        return ret


layers: list[Layer] = []

for n in range(4):
    im = iio.imread(path.join(map_path, f"{n + 1}.png"))
    alpha_mask = im[:, :, -1] > 0
    red_channel = im[:, :, 0]
    layers.append(Layer(red_channel * alpha_mask))
