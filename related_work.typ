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
scalar for describing the !v of !ns.
The !nn subsequently also only predicts scalar !vs.
@azero

!Mp !az drops the assumption of the !g being !zsum by extending the search !vs to vectors.

Specifically, an $n$<join-right> !pl !g returns a score vector with $n$<join-right>
components, where the component at index $i$ denotes the indvidual outcome for !pl $i$.
Likewise, !vs predicted by the !nn and in search tree !ns are also extended to vectors.
MCTS backpropagation is performed with the !v vectors, updating all components in each
visited !n.

Naturally, the !mcts rotates over the !pls in turn order.
When selecting a !n's children, the !algo seeks to maximize component $i$ of the !v
vector, where $i$ denotes the !pl currently at turn.
This !algo is known as $max^n$ search @mcts_survey and similar to the !mp !bi introduced
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
!env, also called _samples_.
Reducing the number of samples is desirable, as it often decreases training time and/or
computational requirements.
To achieve this increase in !sampeff they propose three changes:

First, they notice that the !dnet should map to same !latrep as the !rnet for the same
!g !ss.
By adding a similarity term to the loss !fn, these !net outputs are encouraged to
converge.
This provides a richer training signal since the !latrep is usually a very wide tensor.
@effzero

Compare this to the default !mz !impl:
The !dnet receives training !i only from the single scalar loss !r and any gradients that
flow through the !pnet from the !v and !p losses.
In fact, the training of all neural !nets is driven only by two scalar losses and the
low-dimensional !p loss.
Additionally, it is not even guaranteed that the !dnet and the !rnet agree on the same
representation for identical !g !ss.

#let sscl = [Self-Supervised Consistency Loss]

#[
  #let St = $S_t$
  #let St1 = $S_(t+1)$
  #let Sht1 = $accent(S, hat)_(t+1)$

  // TODO: add image?
  In !mz, the current and next !g !obss $O_t$ and $O_(t+1)$ are fed through the !rnet,
  yielding the !latrep #St respectively #St1.
  From #St and the corresponding !a $a_t$ the !dnet predicts the !latrep #Sht1 @muzero.
  Since both #St1 and #Sht1 are supposed to represent the same !s of the !g, it makes
  sense to introduce a loss that aligns these !latreps.
  This is the idea behind !effz's similarity loss, which #citet("effzero") call
  #emph(sscl)

  However, they employ a stop-gradient operation in the path of #St1, meaning that
  gradients from the similarity loss are not applied to the !rnet !pred #St1.
  This is due to the fact that they closely modeled their !arch after !simsiam @simsiam, a
  self-supervised framework that learns !latreps for images.
  The authors further justify this decision by treating #St1 as the more accurate
  representation and therefore using it as a target for the !dnet.
  @effzero

  In this thesis, I draw inspiration from this idea, but remove the asymmetry in training
  caused by the stop-gradient operation.
  This increases the !sampeff even more.
  // TODO: add forward reference to relevant section
]

The second change aims to improve the selection of !ns during the !mcts:
!mz selects according to the UCT formula, which involves summing the predicted !rs
#footnote[actually, the !rs are discounted according to the !df #sym.gamma prior to
summing]
for each step in the search tree.
The authors claim that in most cases it is not important at which exact future timestep a
certain !r occurs, only that it eventually occurs.
They argue for this assumption from the way humans reason about !gs, and connect it to the
!s aliasing problem.
@effzero
// TODO: For a more detailed discussion see ...

// TODO: clarify
Their suggestion is to introduce a new neural !net to predict the _sum_ of !rs for a given
!seq of !ss in an end-to-end manner.
This effectively sidesteps the question of _when_ a certain !r occurs.
In the paper, this is called _End-To-End Prediction of the Value Prefix_ and implemented
with a LSTM !arch.
@effzero

Third, the authors propose how to mitigate !offp issues.
These arise when reusing old !g !trajs for training:
A !traj created using an earlier !p can be considered outdated in comparison with the
current state of the !nets.
In other words, the !nets already know how to choose better !as, so the !i of the old !g
!traj is of limited use in training.

The original !mz paper already presented the Reanalyze variant which re-runs the !mcts on
old !trajs with the newest !net parameters to provide more accurate !preds of the !p
@muzero.
The !v target is computed by summing future !rs #footnote[over the n-step horizon], which
are fixed due to the recorded !traj, so it naturally suffers from !offp issues @muzero.
#citet("effzero") propose to only use !rs over a smaller horizon, with the size decreasing
for older !trajs to reduce the !offp divergence.
In addition, the !v target is bootstrapped using a !v !pred for the last !s in the
horizon, which is obtained in !mz Reanalyze using a raw !pred from the !pnet @muzero.
!effz instead runs a full !mcts at the last horizon !s, and uses the !v estimate from the
root !n for bootstrapping.
This estimate is an average of multiple !preds throughout the search tree and is therefore
considered a more accurate value.
@effzero

Since the last two modifications were reported to be not as effective, and I also found
them less relatable than the #sscl, I did not implement them.
