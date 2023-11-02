= Evaluation
<sec-eval>
/*
Hier beschreibst du, wie du deinen Ansatz evaluiert hast. Z.B. beschreiben von
Experimenten oder Nutzerstudien.
*/

#import "thesis.typ": citet
#import "drawings/muzero.typ": dyn

This chapter describes the experiments I perform.
The first part describes the !gs I use as !rl !envs in my experiments.
The second and third part present the setup of the ablation studies I perform to examine
the effect of the proposed symmetric latent loss and the use of !tns in the
search tree, respectively.
The final part presents the setup I use to examine if my !mp modifications are able to
learn in a collaborative !mp !g.

== Training !Envs

I apply my !impl of !mz to two different !gs.
In this section, I describe the !g mechanics and !obs tensors.

=== The !G Catch
<sec-game_catch>

Catch is a simple !sp !g which is primarily designed to test if a !rl system is capable of
learning anything at all.
The !g consists of a two-dimensional grid with a configurable number of columns $c$ and
rows $r$.

At the beginning of the !g, a single block spawns in the top row, in a random column.
This block descends by one row in every round of the game, and the !g ends when it reaches
the bottom row.

The !pl can move another block in the bottom row and has to "catch" the falling block by
moving to the same column.
Movement is possible via three !as, which respectively move the bottom block one column to
the left, right or let it stay in its current position.
The !pl's block always starts in the center column when the !g starts.

The !g has no intermediate !rs, but a single terminal !r in the last round, which is 1 if
the !pl caught the falling block, and -1 otherwise.

@fig-game_catch shows a visualisation of the !g for $c = 5$ and $r = 10$ after three rounds:

#[
#import "drawings/catch.typ": catch

#figure(
  catch,
  caption: [The !g Catch in the third round]
) <fig-game_catch>
]

In my experiments I use a grid size of $c = 5$ and $r = 10$.

==== !OBSs

An !obs in Catch consists of a 2D image with a single plane, representing the !g grid.
The image width and height thus equal the number of rows $r$ and columns $c$,
respectively.
The image has zeros everywhere except for the two blocks, where the corresponding pixel is
set to one.

The !obs includes a second tensor with one element, set to a constant 1.
#footnote[This tensor exists for technical reasons only: It is designed to encode the
current !pl in !mp !gs, but degrades to a single-element constant in !sp !gs.]

=== The !Coll !G Carchess

-//I evaluated my proposed !mp modifications on the !g Carchess.
Carchess is a round-based !coll !mp !g on a grid-like structure.
Core element of the game are a number of lanes that cross at intersections, as well as
traffic lights that can be toggled to control the flow of traffic.

A screenshot of a graphical frontend to the !g is shown in
@fig-carchess_screenshot_tutorial.
Instead of traffic lights, the controls are visualized as barriers in this version, but
they work the same way.

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

==== !OBSs

I use !obss of the following form in the !g Carchess, each !obs consisting of three
tensors:

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

#pagebreak(weak: true)
== Ablation Study: Latent Loss Variants
<sec-eval_ll>

I perform experiments to investigate the effects of the different latent loss variants.
Specifically, I compare these three variants:
- no latent loss, like the original !mz !impl proposed by #citet("muzero")
- latent loss with stop-gradient, like in !effz proposed by #citet("effzero"), see @sec-effzero
- symmetric latent loss, which I propose in @sec-mod_symm_latent_loss

At each training step, I choose the unroll length $K tilde.op "unif"{1, K_max}$ to ensure the
unrolling starts at !tss, too.

I use the !g Catch for this evaluation, as introduced in @sec-game_catch.

Apart from the latent loss weight, I fixed all hyperparameters to the values shown in

#[

#import "common.typ": sci_numbers

#show: sci_numbers

#figure(
  table(
    columns: (auto, auto),
    [*Parameter*], [*Value*],
    [Batch size $B$], [256],
    [Number of latent features], [50],
    [Support size for !v and !r !preds $|F|$], [11],
    [Unroll length $K_max$], [3],
    [Optimizer], [Adam],
    [Weight decay], [1e-5],
    [Learning rate], [1e-3],
    [Loss weight for $ell^v$], [1],
    [Loss weight for $ell^r$], [1],
    [Loss weight for $ell^p$], [1e-2],
    [Loss weight for $ell^w$], [1],
    [Discount factor $gamma$], [0.98],
    [pUCT formula $c_1$], [1.5],
    [pUCT formula $c_2$], [1e3],
    [Dirichlet exploration noise $epsilon$], [0.2],
    [Dirichlet exploration noise $alpha$], [0.5],
    [Number of MCTS iterations per move], [30],
    [Training / Selfplay ratio $E$], [100],
    [Random play steps $R$], [3e3],
    [Number of Steps in Buffer], [1e4],
  ),
  caption: [Hyperparameters of the !nns and MCTS for my evaluation of the latent loss],
) <tbl-ll_hparams>

]

=== !NN !ARCH
<sec-eval_catch_nn>

In this section I describe the !nns I use.
If present, a residual block $r(x)$ consisting of a layer $f(x)$ denotes a skip connection
around the layer:
$ r(x) = x + f(x) $

#[
#import "eval/networks.typ": *

#let latent_size = 50
#let action_size = 5
#let turn_size = 3
#let support_shape = (11, 1)

==== !RNET

All !obs tensors are flattened and concatenated to yield a tensor with 51 elements.
This !obs tensor is processed by the following layers:
- #linear(latent_size)
The !latreps in this !arch have therefore #latent_size dimensions.

==== !DNET

#dnet_input(latent_size, action_size)
- #res(linear(55))
- #res(linear(55))
- #linear(64, act: none)
#dnet_output(latent_size, support_shape, turn_size)

==== !PNET

#pnet_input(latent_size)
- #linear(16)
- #res(linear(16))
- #linear(16, act: none)
#pnet_output((11, 1), action_size)

]
