= Related Work
<related_work>
/*
Hier beschreibst du welche verwandte Arbeiten es gibt und wie deine Arbeit sich davon
abhebt.
Nach Prof. André ist ein related work perfekt, wenn der Leser keines der referenzierten
paper selbst lesen muss um die Ideen darin zu verstehen.
*/

#import "thesis.typ": citet

!mz and its precursors already inspired several other works, some of which are summarized
here.

== !Mp !az

While no !mp version for !mz itself exists, #citet("mp_azero") extended its predecessor
!algo !az to !mp capabilities.

The original !impl of !az heavily relies on the !zsum property for some architectural
simplifications:
The !mcts performs a negamax search (see @sec_negamax for details), which only uses a
scalar for describing the !v of !ns @ab_pruning.
The !nn subsequently also only predicts scalar !vs.
@azero

!Mp !az drops the assumption of the !g being !zsum.
This makes it necessary to extend the scalar quantities used in the !algo to vectors.

Specifically, an $n$<join-right> !pl !g returns a score vector $arrow(z)$:
$ arrow(z) = [z_1, z_2, ... z_n] in RR^n $
Each component $z_i$ denotes the indvidual outcome for !pl $i$.
Likewise, !n !vs in the search tree and !preds by the !nn are extended to vectors
$arrow(v) in RR^n$.

MCTS backpropagation is performed with the vectors.
The computations used to update !n statistics are performed with elementwise vector
operations.
This updates each vector component independently.

Naturally, the !mcts rotates over the !pls in turn order.
When selecting a !n's children, the !algo seeks to maximize component $v_i$ of the !v
vector $arrow(v)$, where $i$ denotes the !pl currently at turn.
This !algo is known as maxn search @mcts_survey and similar to the !mp !bi introduced
in @sec_bi_mp.

They evaluate their work on !mp versions of Connect 4 and Tic-Tac-Toe:
The !nets learn to encode knowledge of the !g into search, indicating that the proposed
!mp strategy works in principle.
Performance-wise the !algo places itself below human experts.


== !effz
<rw_effzero>

!effz by #citet("effzero") is a modification of !mz to achieve similar performance like
the original !algo, but requiring less training data.
The amount of training data is usually measured in the number of interactions with the
!env, also called samples.
To achieve this increase in data- and !sampeff they propose three changes.
The most effective and relevant to this work is the introduction of a latent similarity
loss:

#[

#import "drawings/muzero.typ": rep, dyn, pred, training

They notice that the !dnet #dyn should map to same !latrep as the !rnet #rep for the same
!g !ss.
By adding a similarity term to the loss !fn, these !net outputs are encouraged to
converge.
This provides a richer training signal since the !latrep is usually a very wide tensor.

#let actions = $a_t, a_(t+1), ..., a_(t+n-1)$

Specifically, consider two !g !ss $s_t$ and $s_(t+n)$ with the !as #actions in between.
A !latrep for $s_(t+n)$ can be reached in two ways:\
First, with the !rnet #rep directly: $s_(t+n)^0 = #rep (s_(t+n))$.\
Second, by $s_t^n$ as obtained through $n-1$<join-right> inferences with the !dnet from
initially $s_t$ and the !a !seq #actions:
$ s_t^x = cases(
  #rep (s_t) & "if" x = 0,
  #dyn (s_t^(x-1), a_(t+x-1)) & "else",
) $
The idea of the additional similarity loss is to match $s_t^n$ to $s_(t+n)^0$.
This is illustrated in @fig_effzero_loss.

#figure(
  training(draw_latent_loss: true, draw_pnet: false, draw_reward_loss: false),
  caption: [The latent loss introduced in !effz, indicated by the gray arrows.]
) <fig_effzero_loss>

The authors employ a stop-gradient operation on the side of $s_(t+n)^0$, meaning that
gradients from the similarity loss are not applied to the !rnet #rep.
This is due to the fact that they closely modeled their !arch after !simsiam @simsiam, a
self-supervised framework that learns !latreps for images.
The authors further justify this decision by treating $s_t^n$ as the more accurate
representation and therefore using it as a target for the !dnet's !preds.
In @fig_effzero_loss, this is reflected by the unidirectional gray loss arrows.

]
