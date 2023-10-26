= Evaluation
/*
Hier beschreibst du, wie du deinen Ansatz evaluiert hast. Z.B. beschreiben von
Experimenten oder Nutzerstudien.
*/

== The !Coll !G Carchess

I evaluated my proposed !mp modifications on the !g Carchess.
Carchess is a round-based !coll !mp !g on a grid-like structure.
Core element of the game are a number of lanes that cross at intersections, as well as
traffic lights that can be toggled to control the flow of traffic.

A screenshot of a graphical frontend to the !g is shown in
@fig-carchess_screenshot_tutorial.
Instead of traffic lights, the controls are visualized as barriers in this version.

#figure(
  image(
    "drawings/carchess_screenshot_tutorial.jpg",
    height: 10cm,
  ),
  caption: [Screenshot of the !g Carchess],
) <fig-carchess_screenshot_tutorial>

The course of the lanes and the position of traffic lights is defined by the specific map
the !g is played on.

In each round, all !pls must each toggle one distinct traffic light, so that for $n$ !pls,
exactly $n$ traffic lights are toggled.
After all !pls took their turn, a number of $m$ simulation steps are performed, in which
the cars are advanced along their lanes and new cars may spawn.
Finally, the spawn counts for all lanes are updated, and the next round begins.

In each simulation step, all cars move forward one field, with two exceptions:
- the car is on a field with a red traffic light.
- the next field is already occupied by a waiting car, i.e. the next car cannot move
  because it is blocked by other cars or a traffic light.
The last rule does not always apply in intersections, because cars may collide there.
Specifically, a car entering an intersection only waits for a car already in the
intersection if it is on the same lane or drives in the same direction.

In all other cases, two or more cars may end up in the same field after a simulation step,
which is interpreted as a car crash.
The colliding cars are removed from the game and a negative reward is emitted for each
crashed car.

Each lane also has a spawn counter which indicates the number of cars that may spawn
during the next simulation steps.
Specifically, when the first field of a lane is free after a simulation step, a new car is
placed there if the spawn count is nonzero.
Subsequently, the spawn counter is decremented by one.

At the end of each round, each of the spawn counters are incremented by an individual
random amount $x$ drawn from a uniform Distribution $x tilde.op "unif"{s_min, s_max}$.
The maximum spawn count is bounded by the capacity of the lane times a constant $0<c<1$
which is a setting of the !g.
Note that the spawn counters may be incremented even if no cars spawned during the last
simulation, so the cars waiting to be spawned can "pile up" (up to $s_max$).

The objective of the !pls is to manage the traffic lights to maximize the number of cars
which reach the end of their lane.
Each car reaching the end of their lane yields a positive !r for the !pls.

The specific settings of Carchess I use are listed in @tbl-carchess_settings:

#figure(
  table(
    columns: (auto, auto),
    [*Parameter*], [*Value*],
    [Number of !pls $n$], [2],
    [Number of simulation steps per round $m$], [5],
    [Number of total rounds per !g $r$], [10],
    [Maximum car density on a lane $c$], [0.5],
    [Minimum spawn count per round $s_min$], [0],
    [Maximum spawn count per round $s_max$], [4],
    [Reward for each car reaching the end], [2],
    [Reward for each colliding car], [-1],
  ),
  caption: [Settings of Carchess in my evaluation],
) <tbl-carchess_settings>

The map I use is named "tutorial" and shown in @fig-carchess_screenshot_tutorial.
The map's properties are outlined in @tbl-carchess_map_properties:

#figure(
  table(
    columns: (auto, auto),
    [*Property*], [*Value*],
    [Width $w$], [10],
    [Height $h$], [10],
    [Number of lanes $l$], [4],
    [Maximum lane length $a$], [10],
  ),
  caption: [Properties of the Carchess map "tutorial"],
) <tbl-carchess_map_properties>


== !NN !ARCH

I use the following !nns for the evaluation:

=== !OBS

I use !obss of the following form in the !g Carchess, consisting of three tensors:

The first !obs tensor is a three-dimensional image encoding the position of lanes, cars
and traffic lights on the grid.
The image width and height correspond to the size of the map $(w, h)$ (see
@tbl-carchess_map_properties for specific numbers).
The number of image planes depends on the map and the !g settings.

Specifically, there is one image plane per lane to encode the positions of cars on that
lane.
I use a simple onehot encoding where 1 denotes the presence of a car belonging to the
specific lane, and 0 everywhere else.
Another image plane encodes the traffic lights, where -1 denotes a red light (cars cannot
pass), 1 denotes a green light (cars may pass), and 0 everywhere else.

The spawn counts of the lanes are onehot-encoded over several image planes:
There exists an image plane for every other possible spawn count integer
$x = 0, 1, ..., floor(c (a-2))$, where $a$ is the number of fields in the longest lane.
Specifically, an image plane representing a spawn count of $x$ is first initialized with
zeros everywhere.
Then, for each lane which currently has a spawn count of $x$, a 1 is set in the plane, at
the starting location of the lane (where the cars will spawn).

The total number of image planes $i$ can thus be computed by:
$ i = 2 + l + floor(c (a-2)) $
with $l$ being the number of lanes, and $a$ denotes the number of fields in the longest
lane.
The specific values of $l$, $a$ and $c$ are given in @tbl-carchess_map_properties and
@tbl-carchess_settings.

The other two !obs tensors are one-dimensional onehot-encodings of the !pl currently at
turn and the current round number in the !g, respectively.
Their respective lengths are defined by the game settings $n$ and $r$.
Refer to @tbl-carchess_settings for the specific settings I use.



This chapter is divided into two parts.
The first one examines the effects of the different proposed modifications on top of
!mz on training performance.
For this, an ablation study is chosen so that the effect of individual modifications as
well as groups of modifications can be analyzed.
The second part applies the proposed !algo to a !mp !g.
