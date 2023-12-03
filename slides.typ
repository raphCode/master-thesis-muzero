#import "@preview/slydst:0.1.0": *
#import "substitutions.typ"

#show: slides.with(
  title: "Master's Thesis: MuZero",
  subtitle: "Deep Reinforcement Learning with MuZero: Theoretical Foundations, Variants, and Implementation for a Collaborative Game",
  authors: (),
  layout: "medium",
)

#show: substitutions.subs

#let todo = text.with(fill: red)

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

== !az and !mz

- Outperform state-of-the-art programs
- Learn from scratch using selfplay
- Use !MCTS (MCTS) to plan ahead
  
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

== Move selection with MCTS

- Assume reasonably trained !nns are available
- For every !g !s, use MCTS to find a move:

#todo[
image/animation of MC search tree: build tree incrementally across multiple slides:
- initially: root !n / !g !s
- !net inference to yield initial !p and !v
- select !a, use simulator to get next board !repr
- repeat: 800 iterations
- play most visited !a in root !n
]

== Selfplay and !NN Training

#let pad = box.with(width: 3cm)

- Play !gs using MCTS for both !pls
- Record training data $(s, pi, z)$ for each !g !s
  - #pad[*$s$*: !G !s] Observation image
  - #pad[*$pi$*: MCTS !p] Distribution over !as \
    _Which !as of the root !n were visited by the search?_
  - #pad[*$z$*: !G Outcome] Scalar\
    _Ended the game in a win, loss or draw?_
- Supervised training on this data

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

= More Modifications

== Symmetric Latent Similarity Loss

- Based on !effz
- Additional loss to align !latreps of !dnet and !rnet for same !g !ss
- !effz uses a stop-gradient (align !dnet towards !rnet)
- I propose to remove the stop-gradient (align both !nn towards each other)

== !TNs in the MCTS

- Original !mz may search beyond !g end
- I propose:
  - Predict terminal !g !ss during search
  - Disallow !ns beyond !tss
- Implementation:
  - Add special terminal !pl to the set of !pls
  - Is at turn when the !g terminated
  - Trained on ground-truth labels from the !g simulator

= Experiments

== Ablation Study: Symmetric Latent Similarity Loss

#todo[image here]

== Ablation Study: !TNs

#todo[image here]

== Application to Carchess

#todo[image here]

