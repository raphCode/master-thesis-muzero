#import "@preview/slydst:0.1.0": *
#import "substitutions.typ"

#show: slides.with(
  title: "Master's Thesis: MuZero",
  subtitle: [
    Deep Reinforcement Learning with MuZero:
    Theoretical Foundations, Variants, and Implementation for a Collaborative Game
  ],
  authors: (),
  layout: "medium",
)

#show: substitutions.subs

#let todo = text.with(fill: red)

#let reveal(content, condition: false) = {
  if condition {
    content
  } else {
    hide(content)
  }
}

== Content

#outline()

= Overview

== !mz

- Model-based deep !RL !Algo
- Developed by Google DeepMind
- !Gs: Go, shogi, chess, Atari
- Evolution:
  - !ago
  - !agoz
  - !az
  - !mz

#let overview_slide(show_mcts: false) = [
  #import "drawings/slides.typ": architecture_overview

  == !az and !mz

  - Learn from scratch using selfplay
  #reveal(condition: show_mcts)[
    - Use !MCTS (MCTS) to plan ahead
  ]

  #v(1fr)
  #align(center, architecture_overview(show_mcts))
]

#overview_slide(show_mcts: false)
#overview_slide(show_mcts: true)

= !az

== !NN

- Input: 2D image of !g board
- Residual CNN
- Two output heads:
  - *!v*: Scalar \
    _How good is the current position?_
  - *!p*: Distribution over !as \
    _What are promising moves to try in the search?_

#todo[maybe image here]
== !MCTS

- Builds (!g) tree of possible future !as
- Stochastic !algo: Random samples in the !a space
- Tree grows iteratively:
  + *Selection*: _Find the most urgent node_ \
    Guided by:
    - !net !p !preds
    - exploration
    - exploitation
  + *Expansion*: _Add a new node, query !net_ 
  + *Backpropagation*: _Update statistics in the tree_
- New search tree at every !g !s to find a move
- Search results are used as !p training target \
  *$->$ !P improvement*

== Selfplay and !NN Training

#[
#let row(symbol, description, type) = {
  box(width: 3.5cm)[*#symbol*: #description]
  text(type, gray)
}

- Play !gs using MCTS for both !pls
- Record training data $(s, pi, z)$ for each !g !s
  - #row($s$, [!G !s], [!Obs image])
  - #row($pi$, [MCTS !p], [Distribution over !as]) \
    _Which !as of the root !n were visited by the search?_
  - #row($z$, [!G Outcome], [Scalar]) \
    _Ended the game in a win, loss or draw?_
- Supervised training on this data
]

= !mz

== Summary

- Like !az, but no simulator in the tree search
- Instead: Learns a model of the !env

== !NNs

- 3 !nets:
  - !Rnet: !Obs $->$ !Latrep
  - !Pnet: !Latrep $->$ !P, !V
  - !Dnet: !Latrep, !A $->$ !Latrep, !R

== !MCTS

#todo[Animation that builds search tree incrementally]

== Selfplay

#let pad = box.with(width: 4cm)

- Play !gs using MCTS for both !pls
- Record training data $(s, a, r, pi, G)$ for each !g !s
  - #pad[*$s$*: !G !s] Observation image
  - #pad[*$a$*: !A taken] Onehot tensor
  - #pad[*$r$*: !R experienced] Scalar
  - #pad[*$pi$*: MCTS !p] Distribution over !as
  - #pad[*$G$*: n-step !ret] Scalar

== !NN Training

- Sample $K$ consecutive training steps from buffer
- Start with !obs, then unroll !dnet for $K-1$ steps using !as
- End-to-end learning of !ps, !vs and !rs $K$ steps ahead
- Backpropagation-through-time

#todo[image]

= !MP Modifications

== Goals / Improvements over !mz

- !Mp support
- Arbitrary turn order
- General-sum !gs
- Chance events / Stochasticity

== !MP MCTS

- At each !n: Maximize current !pl's profit
- Requires:
  - Current !pl at turn
    - Predicted by the !dnet
    - Trained on ground-truth labels from the !g simulator
  - Per-!pl !vs and !rs
    - $v, r in RR^n$ for $n$ !pls
    - Trained on ground-truth labels from the !g simulator

== Stochasticity

- Add special chance !pl to the set of !pls
- Is at turn when chance events occur
- !P target are the chance outcome !probs (ground-truth from simulator)
- MCTS selects !as according to predicted !p when the chance !pl is at turn






== Application to Carchess

#todo[image here]
