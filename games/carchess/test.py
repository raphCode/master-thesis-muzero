from os import path
from enum import IntEnum
from typing import *
from functools import cache
from collections.abc import *

import numpy as np
import imageio.v3 as iio
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
        self.reset()

    def reset(self):
        percent_closed = 0.6
        self.closed = self.lights & (rng.random(self.lights.shape) < percent_closed)

    def is_closed(self, pos) -> bool:
        return self.closed[pos]

    def plot(self, plt) -> Any:
        y, x = np.nonzero(self.closed)
        return plt.scatter(x, -y, marker="x", s=200, color="red")


class Layer:
    size: tuple[int, int]
    pos: list[Pos]
    cars: set[int]  # indices of cars along the path
    moveable_cars: set[int]

    def __init__(self, mask: ndarr_bool, prepopulate: float = 0.5):
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

        n = len(self.pos) - 1
        self.cars = set((np.arange(n)[rng.random(n) < prepopulate]))
        self.moveable_cars = set()

    @cache
    def offset_pos(self, pos, offset: int = 1) -> Pos:
        return self.pos[self.index_at(pos) + offset]

    @cache
    def index_at(self, pos) -> Optional[int]:
        try:
            return self.pos.index(pos)
        except ValueError:
            return None

    @cache
    def dir_at(self, pos) -> Dir:
        for d in Dir:
            if d.offset(pos) == self.offset_pos(pos, 1):
                return d

    def is_car(self, pos) -> bool:
        return self.index_at(pos) in self.cars

    def mark_car(self, pos) -> None:
        self.moveable_cars.add(self.index_at(pos))

    def advance_cars(self) -> None:
        self.cars -= self.moveable_cars
        self.cars.update(x + 1 for x in self.moveable_cars)
        self.cars.discard(len(self.pos) - 1)
        self.moveable_cars = set()

    def remove_cars(self, pos: Iterable[Pos]) -> None:
        self.cars.difference_update(map(self.index_at, pos))

    def car_pos_iter(self) -> Iterator[Pos]:
        yield from (self.pos[i] for i in self.cars)

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

    def plot(self, plt, offset: float, **kwargs: Any) -> List[Any]:
        def plot(data):
            y, x = np.asarray(data).T
            return x, -y

        pos = np.array(self.pos) + offset
        line_plots = plt.plot(*plot(pos), **kwargs)
        if self.cars:
            cars = np.array([self.pos[c] for c in self.cars]) + offset
            car_plot = plt.scatter(*plot(cars), **kwargs)
            return line_plots + [car_plot]
        return line_plots


def collision_check(layers: Iterable[Layer]) -> list[Pos]:
    car_map = sum(l.car_map.astype(int) for l in layers)
    return list(zip(*np.nonzero(car_map > 1)))


layers: list[Layer] = []

# f, axes = plt.subplots(3,3)
# ax = iter(axes.flatten())
# grid = next(ax)

fig, grid = plt.subplots()
grid.set_xticks(np.arange(100) + 0.5)  # , labels="")
grid.set_yticks(-np.arange(100) + 0.5)  # , labels="")
grid.grid(color="grey")
grid.set_aspect("equal")

im = iio.imread(path.join(map_path, "tl.png"))
alpha_mask = im[:, :, -1] > 0
tl = TrafficLights(alpha_mask)


n_maps = 4

colors = ["tab:red", "tab:blue", "tab:cyan", "tab:orange"]

for n in range(n_maps):
    im = iio.imread(path.join(map_path, f"{n + 1}.png"))
    alpha_mask = im[:, :, -1] > 0
    red_channel = im[:, :, 0]
    layer = Layer(red_channel * alpha_mask)
    layers.append(layer)

    continue
    m = layer.lane_map * 0.2 + layer.car_map
    next(ax).imshow(m)


def can_move(pos: Pos, layer_id: int, layers: list[Layer], tl: TrafficLights):
    if tl.is_closed(pos):
        return False
    next_pos = layers[layer_id].offset_pos(pos)
    for n, l in enumerate(layers):
        if l.is_car(next_pos):
            same_dir = l.dir_at(next_pos) == layers[layer_id].dir_at(pos)
            same_layer = layer_id == n
            if same_dir or same_layer:
                return can_move(next_pos, n, layers, tl)
            return True  # risk a car crash
    # empty field
    return True


def resolve_collisions(layers):
    coll = collision_check(layers)
    print(coll)
    for l in layers:
        l.remove_cars(coll)


def plot_layers(layers) -> List[Any]:
    ret = [tl.plot(grid)]
    for n, l in enumerate(layers):
        ret.extend(l.plot(grid, offset=(n / n_maps - 0.5) / 2, color=colors[n]))
    return ret


artists = []
artists.append(plot_layers(layers))
while coll := collision_check(layers):
    random_layer = layers[rng.integers(len(layers))]
    random_layer.remove_cars(coll)

for step in range(20):
    artists.append(plot_layers(layers))
    for n, l in enumerate(layers):
        for pos in l.car_pos_iter():
            if can_move(pos, n, layers, tl):
                l.mark_car(pos)
    for l in layers:
        l.advance_cars()
    resolve_collisions(layers)
    if step % 5 == 0:
        tl.reset()

ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=1000)

plt.show()
