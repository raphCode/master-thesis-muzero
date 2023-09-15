from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any, Optional, TypeAlias, cast
from functools import cache
from contextlib import suppress
from collections.abc import Iterable

import numpy as np

from .util import Dir, Pos, ndarr_f64, ndarr_int, ndarr_bool

if TYPE_CHECKING:
    import matplotlib  # type: ignore [import]
    from matplotlib.axes import Axes  # type: ignore [import]

    PlotData: TypeAlias = matplotlib.collections.PathCollection | matplotlib.lines.Line2D


rng = np.random.default_rng()


class TrafficLights:
    """
    The traffic light / barrier layer, stores positions and state of all barriers.
    """

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

    def plot(self, ax: Axes) -> PlotData:
        y, x = np.nonzero(self.closed)
        return ax.scatter(x, -y, marker="x", s=200, color="red")

    @property
    def observation_map(self) -> ndarr_int:
        return self.lights * (self.closed.astype(int) * 2 - 1)


class Layer:
    """
    A single lane / car layer, stores positions of cars.
    """

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
        self.spawn_counter = 0

    def update_spawn_count(self, random_number: int, max_density: float) -> None:
        # This double-counts / overestimates the capacity of fields sharing multiple lanes
        # But I talked to the author of the carchess gymenv, it seems to be intentional
        max_capacity = len(self.pos) - 2  # minus first and last field
        self.spawn_counter = min(
            int(max_capacity * max_density),
            self.spawn_counter + random_number,
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
        if self.car_pos:
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
