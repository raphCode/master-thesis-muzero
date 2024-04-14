# My Master's Thesis

**Deep Reinforcement Learning with MuZero:
Theoretical Foundations, Variants, and Implementation
for a Collaborative Game**

Here is the source code of my custom MuZero implementation in PyTorch.
The code passes mypy, and this should be the case for most commits.

For the thesis PDF document, as well as information on other branches in this repo, see
the [main branch](../../tree/main/).

# My Carchess re-implementation

I applied my MuZero implementation to the collaborative multi-player game "Carchess", a
game developed at my academic chair.
A description and screenshot of the game can be found in the thesis paper.

Near the end, I tried actually applying MuZero to Carchess and realized that the official
gymnasium-based Carchess implementation is unsuited.
This is mostly because gymnasium did not expose the chance events and their choices in a
way that my implementation needs it.
I then wrote a custom implementation of the game in a hurry and used that :)

The code lives in [games/carchess](../../tree/master/games/carchess), and there is also a
demo with a matplotlib-UI available:
```sh
cd games
python -m carchess.demo
```

# Run MuZero

This project uses the [just command runner](https://just.systems), so you have to install
that (or figure out what python command to run from the actual justfile).

```sh
pip install -r requirements.txt
just train-carchess
# or without just:
python main.py mcts=muzero  training=carchess networks=carchess game=carchess
```

Happy reverse-engineering if you want to use this project beyond anything documented in
this file!

## Configuration

MuZero is configured by a bunch of yaml files in
[run_config](../../tree/master/run_config):
- **game**: Game to run and its parameters
- **mcts**: Parameters and function implementations to use during the Monte Carlo Tree
  Search
- **networks**: Class implementations for the 3 neural networks
- **training**: All other algorithm parameters

Under the hood, this project uses [hydra](https://hydra.cc/), so you can use all of its
override syntax when running `main.py` (even when using `just`).
In fact, when running `main.py`, hydra enforces to specify configurations for each of the
4 config groups, as outlined above.

> There are a bunch of dirty hacks in the code around hydra. 
This is because:
> - I seem to have different ideas about an 'elegant configuration framework' than the developers
> - The hydra docs are bad (just like the docs of this thesis' work)

## Modular MCTS

The files in [fn](../../tree/master/fn) implement functions that are used during the MCTS,
specifically they control:
- the choice of child nodes during the MCTS selection phase
- the extraction of the final tree policy
- the selection of an action to play in the game

The system allows to easily exchange and test different / new strategies during the MCTS.
