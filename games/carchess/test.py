from __future__ import annotations

import operator
import itertools
from os import path
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Optional, TypeAlias, cast
from functools import cache, cached_property
from contextlib import suppress
from collections.abc import Iterable

import numpy as np
import imageio.v3 as iio
import matplotlib  # type: ignore [import]
import numpy.typing as npt
import matplotlib.pyplot as plt  # type: ignore [import]
import matplotlib.animation as animation  # type: ignore [import]

if TYPE_CHECKING:
    from matplotlib.axes import Axes  # type: ignore [import]


Pos: TypeAlias = tuple[int, int]
ndarr_bool: TypeAlias = npt.NDArray[np.bool_]
ndarr_int: TypeAlias = npt.NDArray[np.int64]
ndarr_f64: TypeAlias = npt.NDArray[np.float64]
PlotData: TypeAlias = matplotlib.collections.PathCollection | matplotlib.lines.Line2D

rng = np.random.default_rng()


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


class TrafficLights:
    lights: ndarr_bool
    closed: ndarr_bool

    def __init__(self, mask: ndarr_bool, percent_closed: float = 0.5):
        self.lights = mask
        self.reset()

    def reset(self, percent_closed: float = 0.5) -> None:
        self.closed = self.lights & (rng.random(self.lights.shape) < percent_closed)

    def toggle_at(self, pos: Pos) -> None:
        assert self.lights[pos]
        self.closed[pos] ^= True

    def is_closed(self, pos: Pos) -> bool:
        return self.closed[pos]  # type: ignore [no-any-return]

    def plot(self, plt: Axes) -> PlotData:
        y, x = np.nonzero(self.closed)
        return plt.scatter(x, -y, marker="x", s=200, color="red")

    @property
    def observation_map(self) -> ndarr_int:
        return self.lights * (self.closed.astype(int) * 2 - 1)


class Layer:
    size: tuple[int, int]
    pos: list[Pos]
    cars: set[int]  # indices of cars along the path
    moveable_cars: set[int]
    spawn_counter: int

    def __init__(self, mask: ndarr_int):
        def highspot_to_start_info(x: int, y: int) -> tuple[Pos, Dir]:
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

        self.size = tuple(mask.shape)  # type: ignore [assignment]
        highspot_pos = cast(
            tuple[int, int], np.unravel_index(np.argmax(mask), mask.shape)
        )
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
        self.reset()

    def reset(self, prepopulate: float = 0.5) -> None:
        n = len(self.pos) - 1
        self.cars = set((np.arange(n)[rng.random(n) < prepopulate]))
        self.moveable_cars = set()

    def update_spawn_count(
        self, min_spawn: int, max_spawn: int, max_density: float
    ) -> None:
        max_capacity = len(self.pos) - 2  # minus first and last field
        self.spawn_counter = min(
            int(max_capacity * max_density),
            self.spawn_counter + rng.integers(min_spawn, max_spawn, endpoint=True),
        )

    @cache
    def next_pos(self, pos: Pos, offset: int = 1) -> Pos:
        index = self.index_at(pos)
        assert index is not None
        return self.pos[index + offset]

    @cache
    def index_at(self, pos: Pos) -> Optional[int]:
        with suppress(ValueError):
            return self.pos.index(pos)
        return None

    @cache
    def dir_at(self, pos: Pos) -> Dir:
        for d in Dir:
            if d.offset(pos) == self.next_pos(pos):
                return d
        raise ValueError

    def is_car(self, pos: Pos) -> bool:
        return self.index_at(pos) in self.cars

    def mark_car(self, pos: Pos) -> None:
        index = self.index_at(pos)
        assert index is not None
        self.moveable_cars.add(index)

    def advance_cars(self) -> bool:
        self.cars -= self.moveable_cars
        self.cars.update(x + 1 for x in self.moveable_cars)
        goal_index = len(self.pos) - 1
        reached_goal = goal_index in self.cars
        self.cars.discard(goal_index)
        self.moveable_cars = set()
        return reached_goal

    def remove_cars(self, pos: Iterable[Pos]) -> None:
        self.cars.difference_update(map(self.index_at, pos))

    @property
    def car_pos(self) -> tuple[Pos]:
        # add two dummy items to force returning a tuple
        getter = operator.itemgetter(0, 0, *self.cars)
        return getter(self.pos)[2:]  # type: ignore [no-any-return]

    @property
    def lane_mask(self) -> ndarr_bool:
        ret = np.zeros(self.size, dtype=bool)
        x, y = zip(*self.pos)
        ret[x, y] = True
        return ret

    @property
    def car_mask(self) -> ndarr_bool:
        ret = np.zeros(self.size, dtype=bool)
        x, y = zip(*self.car_pos)
        ret[x, y] = True
        return ret

    @property
    def observation_maps(self) -> tuple[ndarr_f64, ndarr_int, ndarr_int]:
        x, y = zip(*self.pos)
        lane_map = np.full(self.size, -1)
        lane_map[x, y] = np.linspace(1, 0, len(self.pos))
        spawn_map = np.zeros(self.size, dtype=int)
        spawn_map[self.pos[0]] = self.spawn_counter
        car_map = self.car_mask.astype(int) * 2 - 1
        return lane_map, car_map, spawn_map

    def plot(self, ax: Axes, offset: float, **kwargs: Any) -> list[PlotData]:
        def coords(pos: ndarr_f64) -> tuple[ndarr_f64, ndarr_f64]:
            y, x = pos.T
            return x, -y

        pos = np.array(self.pos) + offset
        line_plots = ax.plot(*coords(pos), **kwargs)
        if self.cars:
            cars = np.array(self.car_pos) + offset
            car_plot = ax.scatter(*coords(cars), **kwargs)
            return line_plots + [car_plot]  # type: ignore [no-any-return]
        return line_plots  # type: ignore [no-any-return]


class Map:
    """
    Map with layers, traffic lights and cars. Provides simulation interface and access to observation tensors.
    """

    layers: list[Layer]
    tl: TrafficLights

    def __init__(self, map_path: str):
        # rounds:int = 20,actions_per_round:int=2, reward_car:float=2, reward_crash :float=-1
        def read_map_img(name: str) -> ndarr_int:
            return iio.imread(path.join(map_path, name))

        def alpha_mask(img: ndarr_int) -> ndarr_bool:
            return img[:, :, -1] > 0

        self.tl = TrafficLights(alpha_mask(read_map_img("tl.png")))
        shape = self.tl.lights.shape
        self.layers = []
        with suppress(FileNotFoundError):
            for layer_id in itertools.count(1):
                img = read_map_img(f"{layer_id}.png")
                assert (
                    img.shape[:2] == shape
                ), f"Shape of layer {layer_id} map {img.shape[:2]} does not match the shape of the traffic light map {shape}"
                red_channel = img[:, :, 0]
                l = Layer(red_channel * alpha_mask(img))
                self.layers.append(l)

    def reset(self, prepopulate: float = 0.3, init_red_rate: float = 0.5) -> None:
        self.tl.reset(init_red_rate)
        for layer in self.layers:
            layer.reset(prepopulate)
        while True:
            coll_pos, _ = self._get_collisions()
            if not coll_pos:
                break
            random_layer = self.layers[rng.integers(len(self.layers))]
            random_layer.remove_cars(coll_pos)

    def update_spawn_counts(
        self, min_spawn: int, max_spawn: int, max_density: float
    ) -> None:
        for l in self.layers:
            l.update_spawn_count(min_spawn, max_spawn, max_density)

    def get_car_observations(self) -> list[npt.NDArray[Any]]:
        channels: list[npt.NDArray[Any]] = []
        for l in self.layers:
            channels.extend(l.observation_maps)
        return channels

    def get_observation(
        self, car_observations: Optional[list[npt.NDArray[Any]]] = None
    ) -> ndarr_f64:
        if car_observations is None:
            car_observations = self.get_car_observations()
        return np.stack([self.tl.observation_map] + car_observations)  # type: ignore [arg-type, operator]

    @cached_property
    def observation_shape(self) -> tuple[int, int, int]:
        return (1 + 3 * len(self.layers), *self.tl.lights.shape)  # type: ignore [return-value]

    def toggle_tl(self, pos: Pos) -> None:
        self.tl.toggle_at(pos)

    def _get_collisions(self) -> tuple[list[Pos], int]:
        car_map = sum(l.car_mask.astype(int) for l in self.layers)
        crash_map = np.maximum(0, car_map - 1)
        x, y = np.nonzero(crash_map)
        return (list(zip(x, y)), crash_map.sum())

    def simulation_step(self) -> tuple[int, int]:
        # local function with local cache:
        # cache is correctly evicted after each simulation step
        @cache
        def can_move(pos: Pos, layer_id: int) -> bool:
            """
            Compute whether a car can move to its next field.
            Implemented with dynamic programming and recursion.
            """
            if self.tl.is_closed(pos):
                return False
            layer = self.layers[layer_id]
            next_pos = layer.next_pos(pos)
            for n, l in enumerate(self.layers):
                if l.is_car(next_pos):
                    same_dir = l.dir_at(next_pos) == layer.dir_at(pos)
                    same_layer = layer_id == n
                    if same_dir or same_layer:
                        # respect car in next field
                        return can_move(next_pos, n)
                    # risk a car crash
                    return True
            # empty field
            return True

        # check moveability
        for n, l in enumerate(self.layers):
            for pos in l.car_pos:
                if can_move(pos, n):
                    l.mark_car(pos)
        # move cars
        goal_cars = 0
        for l in self.layers:
            goal_cars += l.advance_cars()
        # check collisions
        coll_pos, crashed_cars = self._get_collisions()
        for l in self.layers:
            l.remove_cars(coll_pos)
        return goal_cars, crashed_cars

    def plot_grid(self, ax: Axes) -> None:
        ax.set_xticks(np.arange(100) + 0.5)  # , labels="")
        ax.set_yticks(-np.arange(100) + 0.5)  # , labels="")
        ax.grid(color="grey")
        ax.set_aspect("equal")

    def plot(self, ax: Axes) -> list[PlotData]:
        colors = ["tab:red", "tab:blue", "tab:cyan", "tab:orange"]
        ret = [self.tl.plot(ax)]
        for n, l in enumerate(self.layers):
            ret.extend(
                l.plot(ax, offset=(n / len(self.layers) - 0.5) / 2, color=colors[n])
            )
        return ret


m = Map("maps/map1")
m.reset()

fig, ax = plt.subplots()

m.plot_grid(ax)

artists = []

for step in range(20):
    artists.append(m.plot(ax))
    m.simulation_step()

ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=1000)

plt.show()
