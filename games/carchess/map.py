from __future__ import annotations

import itertools
from os import path
from string import ascii_lowercase, ascii_uppercase
from typing import TYPE_CHECKING, Any, Optional, TypeAlias, cast
from pathlib import Path
from functools import cache, cached_property
from contextlib import suppress

import numpy as np
import imageio.v3 as iio
import numpy.typing as npt

from .layers import Layer, TrafficLights

if TYPE_CHECKING:
    import matplotlib  # type: ignore [import]
    from matplotlib.axes import Axes  # type: ignore [import]

    PlotData: TypeAlias = matplotlib.collections.PathCollection | matplotlib.lines.Line2D

    from .util import Pos, ndarr_f64, ndarr_int, ndarr_bool

rng = np.random.default_rng()


class Map:
    """
    Map with layers, traffic lights and cars.
    Provides a simulation interface and access to observation tensors.
    """

    layers: list[Layer]
    tl: TrafficLights

    def __init__(self, map_name_or_path: str):
        def read_map_img(name: str) -> ndarr_int:
            default_map_path = Path(__file__).parent / "maps"
            return iio.imread(path.join(default_map_path, map_name_or_path, name))

        def alpha_mask(img: ndarr_int) -> ndarr_bool:
            return img[:, :, -1] > 0

        self.tl = TrafficLights(alpha_mask(read_map_img("tl.png")))
        self.layers = []
        with suppress(FileNotFoundError):
            for layer_id in itertools.count(1):
                img = read_map_img(f"{layer_id}.png")
                assert img.shape[:2] == self.size, (
                    f"Size of layer {layer_id} map {img.shape[:2]} "
                    f"does not match the size of the traffic light map {self.size}"
                )
                max_color = img[:, :, :-1].max(axis=-1)
                l = Layer(max_color * alpha_mask(img))
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

    def update_spawn_count_explicit(
        self,
        layer_id: int,
        random_number: int,
        max_density: float,
    ) -> None:
        self.layers[layer_id].update_spawn_count(random_number, max_density)

    def update_spawn_counts_random(
        self,
        min_spawn: int,
        max_spawn: int,
        max_density: float,
    ) -> None:
        random_numbers = rng.integers(
            min_spawn, max_spawn, endpoint=True, size=len(self.layers)
        )
        for l, n in zip(self.layers, random_numbers):
            l.update_spawn_count(n, max_density)

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
        return np.stack(
            [self.tl.observation_map] + car_observations,
        )

    @cached_property
    def observation_shape(self) -> tuple[int, int, int]:
        return (1 + 3 * len(self.layers), *self.tl.lights.shape)  # type: ignore [return-value] # noqa: E501

    @cached_property
    def size(self) -> tuple[int, int]:
        return cast(tuple[int, int], self.tl.lights.shape)

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
            l.maybe_spawn_car()
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

    def ascii_art(self) -> str:
        field = np.full(self.size, " ", dtype=str)
        for n, layer in enumerate(self.layers):
            # x, y = zip(*layer.car_pos)
            field[self.tl.lights] = "o"
            field[self.tl.closed] = "x"
            for x, y in layer.car_pos:
                s = ascii_lowercase if self.tl.closed[x, y] else ascii_uppercase
                field[x, y] = s[n]
            field[*layer.pos[0]] = layer.spawn_counter
        return np.array_str(field)
