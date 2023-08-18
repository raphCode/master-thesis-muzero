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
ndarr_int: TypeAlias = npt.NDArray[np.int]

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

    def reset(self, percent_closed:float=0.5):
        self.closed = self.lights & (rng.random(self.lights.shape) < percent_closed)

    def toggle_at(self,pos)->None:
        assert self.lights[pos]
        self.closed[pos] ^= True

    def is_closed(self, pos) -> bool:
        return self.closed[pos]

    def plot(self, plt) -> Any:
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
    spawn_counter:int

    def __init__(self, mask: ndarr_bool):
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
        self.reset()

    def reset(self, prepopulate: float=0.5)->None:
        n = len(self.pos) - 1
        self.cars = set((np.arange(n)[rng.random(n) < prepopulate]))
        self.moveable_cars = set()
    def update_spawn_count(self,min_spawn:int, max_spawn:int, max_density:float)->None:
        max_capacity = len(self.pos) -2 # minus first and last field
        self.spawn_counter=min(int(max_capacity*max_density),self.spawn_counter+ rng.integers(min_spawn, max_spawn, endpoint=True))

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

    def advance_cars(self) -> bool:
        self.cars -= self.moveable_cars
        self.cars.update(x + 1 for x in self.moveable_cars)
        goal_index =len(self.pos)-1
        reached_goal = goal_index in self.cars
        self.cars.discard(goal_index)
        self.moveable_cars = set()
        return reached_goal

    def remove_cars(self, pos: Iterable[Pos]) -> None:
        self.cars.difference_update(map(self.index_at, pos))

    def car_pos_iter(self) -> Iterator[Pos]:
        yield from (self.pos[i] for i in self.cars)

    @property
    def lane_mask(self) -> ndarr_bool:
        ret = np.zeros(self.size, dtype=bool)
        for pos in self.pos:
            ret[pos] = True
        return ret

    @property
    def car_mask(self) -> ndarr_bool:
        ret = np.zeros(self.size, dtype=bool)
        for index in self.cars:
            ret[self.pos[index]] = True
        return ret

    @property
    def observation_map(self) -> ndarr_int:
        x,y=zip(*self.pos)
        lane_map = np.full(self.size, -1)
        lane_map[x,y] = np.linspace(1, 0, len(self.pos))
        spawn_map = np.zeros(self.size)
        spawn_map[self.pos[0]] =self.spawn_counter
        car_map = self.car_mask.astype(int) * 2 - 1
        return np.stack(lane_map,car_map, spawn_map)

    def plot_grid(self, ax)->Any:
        ax.set_xticks(np.arange(100) + 0.5)  # , labels="")
        ax.set_yticks(-np.arange(100) + 0.5)  # , labels="")
        ax.grid(color="grey")
        ax.set_aspect("equal")

    def plot(self, plt, offset: float, **kwargs: Any) -> List[Any]:
        def coords(pos:ndarr_f64):
            y, x = pos.T
            return x, -y

        pos = np.array(self.pos) + offset
        line_plots = plt.plot(*coords(pos), **kwargs)
        if self.cars:
            cars = np.array([self.pos[c] for c in self.cars]) + offset
            car_plot = plt.scatter(*coords(cars), **kwargs)
            return line_plots + [car_plot]
        return line_plots



layers: list[Layer] = []

# f, axes = plt.subplots(3,3)
# ax = iter(axes.flatten())
# grid = next(ax)

fig, ax = plt.subplots()

im = iio.imread(path.join(map_path, "tl.png"))
alpha_mask = im[:, :, -1] > 0
tl = TrafficLights(alpha_mask)


n_maps = 4


for n in range(n_maps):
    im = iio.imread(path.join(map_path, f"{n + 1}.png"))
    red_channel = im[:, :, 0]
    layer = Layer(red_channel * alpha_mask)
    layers.append(layer)

    continue



class Map:
    layers:list[Layer]
    tl:TrafficLights

    def __init__(self, map_path: str):
        # rounds:int = 20,actions_per_round:int=2, reward_car:float=2, reward_crash :float=-1
        def read_map_img(name:str)->ndarr_int:
            return iio.imread(path.join(map_path,name))
        def alpha_mask(img:ndarr_f64)->ndarr_bool:
            return img[:, :, -1] > 0
        self.tl=TrafficLights(alpha_mask(read_map_img( "tl.png") ))
        
            

    def reset(self, prepopulate: float=0.3,  init_red_rate:float=0.5)->None:
        for layer in self.layers:
            layer.reset(prepopulate)
        self.tl.reset(init_red_rate)

    def update_spawn_counts(min_spawn:int, max_spawn:int, max_density:float)->None:
        # car_density:float=0.5, spawn_rate:tuple[int,int] 
        for l in self.layers:
            l.update_spawn_count(min_spawn, max_spawn, max_density)

    def toggle_tl(self,pos)->None:
        self.tl.toggle_at(pos)

    def simulation_step()->tuple[int,int]:
        # local function with local cache:
        # cache is correctly evicted after each simulation step
        @cache
        def can_move(pos: Pos, layer_id: int)->bool:
            """
            Compute whether a car can move to its next field.
            Implemented with dynamic programming and recursion.
            """
            if self.tl.is_closed(pos):
                return False
            layer = self.layers[layer_id]
            next_pos = layer.offset_pos(pos)
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

        def collision_check() -> tuple[list[Pos], int]:
            car_map = sum(l.car_mask.astype(int) for l in self.layers)
            crash_map = np.max(0, car_map-1)
            return list(zip(*np.nonzero(crash_map))), crash_map.sum()

        # check moveability
        for n, l in enumerate(layers):
            for pos in l.car_pos_iter():
                if can_move(pos, n, layers, tl):
                    l.mark_car(pos)
        # move cars
        goal_cars = 0
        for l in self.layers:
            goal_cars += l.advance_cars()
        # check collisions
        coll_pos, crashed_cars = collision_check(layers)
        for l in self.layers:
            l.remove_cars(coll_pos)
        return goal_cars, crashed_cars

    def plot(self, plt) -> List[Any]:
        colors = ["tab:red", "tab:blue", "tab:cyan", "tab:orange"]
        ret = [tl.plot(plt)]
        for n, l in enumerate(self.layers):
            ret.extend(l.plot(plt, offset=(n / len(self.layers) - 0.5) / 2, color=colors[n]))
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
