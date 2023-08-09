from os import path
from enum import IntEnum
from typing import *
from functools import cache
from collections.abc import *

import numpy as np
import imageio.v3 as iio
import numpy.typing as npt
import matplotlib.pyplot as plt

map_path = "maps/map1"

Pos: TypeAlias = tuple[int, int]
ndarr_bool: TypeAlias = npt.NDArray[np.bool_]

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


class TrafficLights:
    lights: ndarr_bool
    closed: ndarr_bool

    def __init__(self, mask: ndarr_bool, percent_closed: float = 0.5):
        self.lights = mask
        self.closed = mask & (rng.random(mask.shape) < percent_closed)

    def is_closed(self, pos) -> bool:
        return self.closed[pos]

    def plot(self, plt) -> None:
        y, x = np.nonzero(self.closed)
        plt.scatter(x, -y, marker="x", s=200, color="red")


class Layer:
    size: tuple[int, int]
    pos: list[Pos]
    cars: set[int]  # indices of cars along the path

    def __init__(self, mask: ndarr_bool, prepopulate: float = 0.2):
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

    def advance_cars(self, car_indices: Iterable[int]) -> None:
        n = len(self.pos)
        assert list(sorted(self.cars)) == self.cars
        for i in car_indices:
            self.cars[i] += 1
        if self.cars[-1] > n:
            del self.cars[-1]

    def remove_cars(self, car_indices: Iterable[int]) -> None:
        for i in sorted(car_indices, reverse=True):
            del self.cars[i]

    @property
    def lane_map(self) -> ndarr_bool:
        ret = np.zeros(self.size, dtype=bool)
        for pos in self.pos:
            ret[pos] = True
        return ret

    @property
    def car_map(self) -> ndarr_bool:
        ret = np.zeros(self.size, dtype=bool)
        for index in self.cars:
            ret[self.pos[index]] = True
        return ret

    def plot(self, plt, offset: float, **kwargs: Any) -> None:
        def plot(data):
            y, x = np.asarray(data).T
            return x, -y

        pos = np.array(self.pos) + offset
        cars = np.array([self.pos[c] for c in self.cars]) + offset
        plt.plot(*plot(pos), **kwargs)
        plt.scatter(*plot(cars), **kwargs)


def collision_check(layers: Iterable[Layer]) -> list[Pos]:
    car_map = sum(l.car_map.astype(int) for l in layers)
    return list(zip(*np.nonzero(car_map > 1))), car_map > 1


layers: list[Layer] = []

# f, axes = plt.subplots(3,3)
# ax = iter(axes.flatten())
# grid = next(ax)

f, grid = plt.subplots()
grid.set_xticks(np.arange(100) + 0.5, labels="")
grid.set_yticks(-np.arange(100) + 0.5, labels="")
grid.grid(color="grey")
grid.set_aspect("equal")

im = iio.imread(path.join(map_path, "tl.png"))
alpha_mask = im[:, :, -1] > 0
tl = TrafficLights(alpha_mask)

tl.plot(grid)

n_maps = 4

colors = ["tab:red", "tab:blue", "tab:cyan", "tab:orange"]

for n in range(n_maps):
    im = iio.imread(path.join(map_path, f"{n + 1}.png"))
    alpha_mask = im[:, :, -1] > 0
    red_channel = im[:, :, 0]
    layer = Layer(red_channel * alpha_mask)
    layers.append(layer)
    layer.plot(grid, offset=(n / n_maps - 0.5) / 2, color=colors[n])

    continue
    m = layer.lane_map * 0.2 + layer.car_map
    next(ax).imshow(m)


coll, m = collision_check(layers)
# next(ax).imshow(m)
print(coll)


plt.show()
