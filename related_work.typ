= Related Work
<related_work>
/*
Hier beschreibst du welche verwandte Arbeiten es gibt und wie deine Arbeit sich davon
abhebt.
Nach Prof. André ist ein related work perfekt, wenn der Leser keines der referenzierten
paper selbst lesen muss um die Ideen darin zu verstehen.
*/

#import "thesis.typ": citet

muzero and its precursors already inspired several other works, some of which are
summarized here.

== Mp azero  // {{{

muzero was originally implemented for sp games, as well as for 2p 0sum games.
While no mp version for muzero itself exists, #citet("multiplayer_alphazero")
extended the predecessor algo azero to mp capabilities.

Importantly, they relax the assumption of the game being 0sum:
A 0sum game involves two players where one player's gain comes with an equivalent loss for
the other player.
The azero (and in turn muzero) algo make use of this fact by directly predicting how good
the current position is for the current player by a scalar val output of the pnet
#cite("alphazero", "muzero").
The mp extension predicts a val vector instead, which provides an estimate of the expected
individual utilities for all players at the same time.

Likewise, they also extend the game to return scores for each player at the end instead of
a single outcome.
Naturally, the algo rotates over the list of all players instead of alternating between
two players.~@alphazero
They evaluate their work on mp versions of Connect 4 and Tic-Tac-Toe:
The nets learn to encode knowledge of the game into search, indicating that the proposed
mp strategy works in principle.
Performance-wise the algo places itself below human experts.

// }}}
== effzero  // {{{
<rw_effzero>

effzero by #citet("effzero") is a modification of muzero to achieve similar performance
like the original algo, but requiring less training data.
The amount of training data is usually measured in the number of interactions with the
environment, also called _samples_.
Reducing the number of samples is desirable, as it often decreases training time and/or
computational requirements.
To achieve this increase in sampeff they propose three changes:

First, they notice that the dynamics model should map to same latrep as
the rnet for the same game states.
By adding a similarity term to the loss function, these net outputs are encouraged to
converge.
This provides a richer training signal since the latrep is usually a very
wide tensor.~@effzero

Compare this to the default muzero impl:
The dnet receives training information only from the single scalar loss rew and any
gradients that flow through the pnet from the val and pol losses.
In fact, the training of all neural nets is driven only by two scalar losses and the
low-dimensional pol loss.
Additionally, it is not even guaranteed that the dnet and the rnet agree on the same
representation for identical game states.

#let sscl = [_Self-Supervised Consistency Loss_]

#[
  #let St = $S_t$
  #let St1 = $S_(t+1)$
  #let Sht1 = $accent(S, hat)_(t+1)$

  // TODO: add image?
  In muzero, the current and next game obss $O_t$ and $O_(t+1)$ are fed through the rnet,
  yielding the latrep #St respectively #St1.
  From #St and the corresponding action $a_t$ the dnet predicts the latrep #Sht1.~@muzero
  Since both #St1 and #Sht1 are supposed to represent the same state of the game, it makes
  sense to introduce a loss that aligns these latreps.
  This is the idea behind effzero's similarity loss, which #citet("effzero") call #sscl.

  However, they employ a stop-gradient operation in the path of #St1, meaning that
  gradients from the similarity loss are not applied to the rnet pred #St1.
  This is due to the fact that they closely modeled their architecture after
  simsiam~@simsiam, a self-supervised framework that learns latreps for images.
  The authors further justify this decision by treating #St1 as the more accurate
  representation and therefore using it as a target for the dnet.~@effzero
  In this thesis, I draw inspiration from this idea, but remove the asymmetry in training
  caused by the stop-gradient operation.
  This increases the sampeff even more.
  // TODO: add forward reference to relevant section
]

The second change aims to improve the selection of nodes during the mcts:
muzero selects according to the UCT formula, which involves summing the predicted rews
#footnote[actually, the rews are discounted according to the df factor #sym.gamma]
for each step in the search tree.
The authors claim that in most cases it is not important at which exact future timestep a
certain rew occurs, only that it eventually occurs.
They argue for this assumption from the way humans reason about games, and connect it to
the state aliasing problem.~@effzero
// TODO: For a more detailed discussion see ...

// TODO: clarify
Their suggestion is to introduce a new neural net to predict the _sum_ of rews for a given
sequence of states in an end-to-end manner.
This effectively sidesteps the question of _when_ a certain rew occurs.
In the paper, this is called _End-To-End Prediction of the Value Prefix_ and implemented
with a LSTM architecture.~@effzero

Third, the authors propose how to mitigate offpol issues.
These arise when reusing old game trajs for training:
A traj created using an earlier pol can be considered outdated in comparison with the
current state of the nets.
In other words, the nets already know how to choose better actions, so the information of
the old game traj is of limited use in training.

The original muzero paper already presented the Reanalyze variant which re-runs the MCTS
on old trajs with the newest net parameters to provide more accurate predictions of
the pol~@muzero.
The vtar is computed by summing future rews#footnote[over the nsh], which are fixed due to
the recorded traj, so it naturally suffers from offp issues~@muzero.
#citet("effzero") propose to only use rews over a smaller horizon, with the size
decreasing for older trajs to reduce the offpol divergence.
In addition, the vtar is bootstrapped using a val pred for the last state in the horizon,
which is obtained in muzero Reanalyze using a raw pred from the pnet~@muzero.
effzero instead runs a full MCTS at the last horizon state, and uses the val
estimate from the root node for bootstrapping.
This estimate is an average of multiple predictions throughout the search tree and is
therefore considered a more accurate value.~@effzero

Since the last two modifications were reported to be not as effective, and I also found
them less relatable than the #sscl, I did not implement them.

// }}}
