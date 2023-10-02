= Approach
<approach>
/*
Hier beschreibst du, was du in der Arbeit neues gemacht hast, und wie du es implementiert
hast.
*/

#import "thesis.typ": citet, blockquote
#import "drawings/muzero.typ": rep, dyn, pred


The original !mz !impl is limited to a specific class of !gs.
I begin by reviewing the limitations of !mz and the reasons behind them, and then move on
to a discussion of how I extend the !arch to more general !gs.

== !mz Limitations
<sec_muzero_limitations>

The !impl of !mz is designed for single-!ag !envs and !2p !zsum !gs.
Furthermore, all !gs are expected to be deterministic and of !pinf.
The causes of these limitations and some implications are briefly discussed below.

=== Determinism

In order to plan ahead, the future !s of the !env must be predictable for an intial !s
$s^0$ and given !seq of !as.
The !dnet #dyn in !mz is a deterministic !fn and no chance !ss are modeled in the !arch.
Perhaps unexpectedly, #citet("stochastic_muzero") show that !mz's performance falls short
in a stochastic !env compared to other methods that model stochasticity.

=== !PINF

Accurately planning ahead also relies on unambiguously identifying the initial state
$s^0$.
From a !gtic standpoint, this requires the !g to be with !pr and of !pinf (See @sec_pr and
@sec_pinf, respectively).
In the context of !rl, it also means that an !obs must uniquely capture the current !s of
the !env.

This aspect is best illustrated by the example of the fifty-moves rule in chess:
If 50 moves pass without a capture or a pawn moved, the !g may end in a draw.
While a human can deduce a history of moves from successive board !ss, the !mz !ag starts
each move afresh, given only the current !obs.
The current board is therefore not enough !i to distinguish a regular !g situation from
one where the fifty-move rule applies.
@muzero

=== !ZSUM !Gs

In the !2p setting, !mz assumes that the !g is !zsum.
This assumption is built into the !arch itself, because it performs negamax search (see
@sec_negamax) during MCTS @muzero.

Negamax exploits the !zsum property by using only a single scalar for a !n's !v.
The design of the !nns (#dyn<join-right> and #pred) in !mz follows this choice and also
only predict a single scalar for !s !vs and transition !rs.

=== Fixed Turn Order

The limitation to !gs with up to two !pls implicitly makes assumptions about the turn
order.
In !sp !gs, trivially only one !pl can be at turn.

In !gs with 2 !pls, alternating turns are assumed.
This is not a limitation in practice, since the turn order mechanics of any !g can be
modeled with the set of available !as $A$.

As an contrived example, consider castling in chess:
It may be viewed as two consecutive turns of the same !pl, moving king and rook
separately.
However, by expanding the !a set $A$ with a castling move, the assumption about
alternating turn order still holds.

The alternating turn order is exploited by the negamax search !impl.
For single-!ag !envs, negamax is disabled altogether.
@muzero

== Extension to !MP, Stochastic and General Sum !Gs

I propose an extension of !mz to more general !gs than the original !impl is capable of.
This includes !gs with chance events, more than two !pls, and therefore arbitrary payoffs.

Planning ahead in a !g with more than one !pl requires some !i or assumptions about the
behavior of other !pls.
In a !2p !zsum !g, the behavior of the opponent !pl is easy to model:
He will always try to minimize the score of the other !pl.
This assumption does not hold for general-sum !gs with arbitrary payoffs.

My extension of !mz to !mp !gs is inspired from the !mp !bi in !gt.
It assumes a non-cooperative setting, where each !pl tries to maximize his individual
payoff, like introduced in @sec_bi_mp.

I perform a number of changes to the !algo.
The updated training setup with all modifications is visualized in @fig_raphzero_training.

=== Individual !Vs

#[
#let vector(x) = $arrow(#x) = [#x _1, #x _2, ..., #x _n] in RR^n$

!Mp !bi requires to keep track of the individual !exuts and !rs for each !pl.
I follow the design of !mp !az @mp_azero (see @sec_mp_azero) and replace all scalars
describing an !env !r, !s or !n !v, with vectors.
In an !env with $n$ !ags, these !r and !v vectors consist of $n$ components:
$ vector(r) \
  vector(v) $
Each component $r_i$ and $v_i$ denotes the individual !r and !v of !ag $i$, respectively.

#let (r1, r2, r3) = (2, -1, 0)

For example, the !r $arrow(r) = [r1, r2, r3]$ indicates that the first !ag was rewarded
with~#r1, the second !ag received a !r of~#r2, and the third !ag got no !r.
#assert(r3 == 0)
]

Note that in a collaborative !g, all individual !rs are shared, as outlined in
@sec_gt_collab:
$ r_i = r "for" 1 <= i <= n $

This modification is reflected in @fig_raphzero_training by adding vector arrows
$arrow(x)$ to !mp data.

=== Turn Order !Pred
<sec_mod_turn>

Making informed decisions within the search tree requires not only individual !rs and !vs,
but also an understanding who can make a decision at a particular !n.
In !mz, the turn order is hardcoded by using negamax search in !2p !gs.

To achieve a more general !algo, my !impl does not make any assumptions about the turn
order.
Instead, the next !pl at turn is learnt by the !dnet #dyn.

I added an additional output head $w$ to the !dnet #dyn:
$ (s^n, r^n, w^n) = #dyn (s^(n-1), a^(n-1)) $
which predicts the !pl $w^n$ at turn in !s
$s^n$.
It is implemented as a categorical distribution $T$ over the set of !pls $W$:
$ T(s, w) = Pr(w|s) $

During MCTS, for each !s $s^n$ encountered, the current !pl $w^n$ is assumed to be the one
with the highest predicted !prob:
$ w^n = limits("argmax")_(w in W) ( T(s^n, w) ) $

The turn output $w$ is trained like the !r $arrow(r)$, based on ground-truth labels given
by the !g simulator during selfplay.

@fig_raphzero_training shows this change by the added $w$ in the !env transitions and
!dnet !preds.

=== maxn !MCTS

Following !mp !bi (@sec_bi_mp), the MCTS selection phase considers !n !vs for the !pl currently at turn
only.
Specifically, each !n !v is a vector:
$ arrow(Q)(s^n, a^n) = gamma arrow(v)^(n+1) + arrow(r)^(n+1) $
Let $Q_i (s, a)$ denote the $i$<no-join>-th component of this vector.

In !s $s^k$, maxn-MTCS then selects an !a $a^k$ as to maximize $Q_i (s^k, a^k)$ where $i =
w^k$, the !pl currently at turn, as outlined in @sec_mod_turn:
$ a^k = limits("argmax")_a ( arrow(Q)_w_i (s, a) + u(s, a) ) $
where $u(s, a)$ represents some bonus term to incorporate exploration and the prior !probs
$P(s^k, a^k)$ into the decision.

=== Chance events

I model stochastic !envs with an explicit chance !pl.
He is at turn whenever a chance event occurs in the !g.

#[
#let wc = $w_frak(C)$

The occurrence of chance events is given by the !dnet as part of the turn order !pred $w$.
An additional special !pl #wc is added to the set of !pls $W$:
$ W' = W union {wc} $

If $w^n = #wc$, the current decision !n $s^n$ is assumed to be a chance event.
The !probs of the different chance outcomes are predicted by the !pnet #pred.
During MCTS, child !ns of a chance event $s^n$ are selected solely according to the !p $p^n$.

Like the current !pl at turn, the occurrence of chance events is trained on ground-truth
labels given by the !g simulator.
The !g simulator also provides the exact chance outcomes $c_t$ if !s $s_t$ is a chance
event.
These outcomes are used as targets during training for the !p $p$ as predicted by #pred.

Chance events are represented by a dice in @fig_raphzero_training
#footnote[ignore the fact that the !g !ttt actually has no chance events].
Note that a chance event $s_t$ differs from regular !g !ss in the figure in two aspects:
- the !p target for training #pred are the chance outcomes $c_t$
- the target for the turn order !preds $w$ is the constant chance !pl #wc

#import "drawings/muzero.typ": rep, dyn, pred, training

#let chance_state = 1
#let t(n) = if n == -1 { $T$ } else { $t+#n$ }

#figure(
  training(
    dynamics_env: n => {
      (
        $ arrow(r)_#t(n) $,
        if (n - 1) != chance_state { $ w_#t(n) $ } else { $ wc $ },
      )
    },
    dynamics_net: n => ($ arrow(r)_t^#n $, $ w_t^#n $),
    use_vectors: true,
    value_target: "return",
    chance_event: chance_state,
  ),
  caption: [Training setup of my !impl of !mz for stochastic multi-agent !envs]
) <fig_raphzero_training>

]

== Further Enhancements

Furthermore, I implemented the following modifications:

=== Symmetric Latent Similarity Loss

As outlined in @rw_effzero, #citet("effzero") already layed the groundwork for
improvements in !sampeff by introducing a similarity loss between !preds of #rep and dyn.
However, I thank that their adoption of the stop-gradient operation from !simsiam @simsiam
may have been short-sighted:

In !simsiam, the task is to learn discriminative !latreps from input data in a
self-supervised manner.
Self-supervised learning may suffer from collapsed solutions, where all learned !reprs end
up being very similar or even identical @ss_decorr.
#citet("simsiam") show that a stop-gradient is effective in preventing collapsed solutions
in their !arch.

However, #citet("ss_decorr") show that it is possible to learn significant !latreps
without the use of a stop-gradient mechanism:
The only requirement is that a decorrelation mechanism of some form must be present in the
!arch that penalizes collapsed solutions.

In !mz, I hypothesize that a decorrelation is already achieved by the training losses
$ell^r$, $ell^p$<join-right> and $ell^v$.

Intuitively, if the !latreps of #rep and #dyn were to collapse, the !r, !p and !v !preds
can not match their targets.
Subsequently, the training loss would stay high.
In order to accurately predict these quantities, #pred<join-right> and #dyn must encode
useful !i in the latent space from the !obs and !seq of !as, respectively.

Consequently, I see no risk of latent collapse in the !effz !arch, and propose to remove
the stop-gradient operation.

=== !TNs in the !MCTS

In !az, the !mcts uses a perfect simulator to determine the next !g !s for hypothetical
!as.
Naturally, this simulator also indicates when the !g is over and there are no further
moves to search for @azero.
These !tss are an important concept in !RL, since their value is by definition zero:
no future !rs can occur @sutton.
Additionally, in !gs, the !r is often sparse, meaning that a non-zero !r occurs only at
the end of the !g.
In this case, the !r of !tss is the only driving force behind learning a good !p.

Likewise, in the !mcts, !tns are important anchors that provide a known, ground-truth !r
and !v:
During search, !rs and !vs of !ns are backpropagated upward along the search path from
children to parents.
Backpropagation from !tns provides upstream !ns with valuable !i that ultimately allows
the !ag to make an informed decision about what !a to take.
In fact, by applying this process iteratively, it is possible to evaluate !as for !g !ss
many steps before the end of the !g.

!mz replaces the perfect simulator with the !dnet, which provides !r !preds for
transitions between !ns.
is unexpected that !mz does not include any concept of !tns:

#blockquote("muzero")[
  _MuZero_ does not give special treatment to terminal nodes and always uses the value
  predicted by the network. Inside the tree, the search can proceed past a terminal node -
  in this case the network is expected to always predict the same value. This is achieved
  by treating terminal states as absorbing states during training.
]

As the search progresses past a !tn, the predicted !rs and !vs past the !tn are
backpropagated in the same way as for any other !n.
However, it's crucial to note that since the !v of a !ts is by definition zero,
backpropagating any non-zero value to a !tn is unsound.
Under the assumption that the !nets are able to accurately predict the terminal !r,
backpropagation effectively renders the !TNs unreliable.

In general, these scenarios are possible regarding !tss:
- Ignore !tss during search and training
- Train zero !v and !r for !ss beyond the end
- Predict the occurrence of !tss:
  - Disallow searching past !tns
  - Only create child !ns with zero !v and !r

Completely ignoring !tns forces the !nets to predict values for !latreps beyond the !g end
they were never trained on, which produces nonsense.
If values of large magnitude are output, it can even lead to numerical instabilities when
operations like softmax are applied to the predictions.

Learning zero !rs and !vs for !ss beyond terminal ones, as !mz does, seems reasonable.
However, it incurs some overhead during training:
The !dnet and !pnet must be unrolled for a number of steps beyond the !tss, which requires
more computational power.
Also, the !nets might fail to generalize beyond the unrolled horizon, simply because they
were never trained that far.
This brings us back to the first scenario, if the tree search continues further beyond the
end of the !g than was anticipated during training.

In this thesis, I chose the third option because it allows !tss to be modeled accurately
during the search without extra computational cost.
I implemented it by adding a scalar output to the !dnet, which is trained to predict
whether the !g ends at the next !s.
Adding the output to the !dnet is favorable over the !pnet, since no additional !g !ss
have to be added during training.
Moreover, if we consider the idea of using the !pnet to classify whether a certain !g !s
is terminal, we have to use a !latrep to encode the !s.
However, a !ts has neither an !obs nor a meaningful !v or !p, so it makes little sense to
associate it with a !latrep#footnote[albeit that would be possible].

Once !tss are predicted, the search behavior can be adjusted.
The simplest approach seems to be to simply abort the search beyond !tns and not allow any
further visits to that !n.
However, this introduces a problem:
!mz builds the !p target using the visit counts of the search !ns.
Blocking the search from continuing past !tns skews these visit counts.
Since !ns are blocked from being visited, other !ns must be selected instead, resulting in
visit counts that do not accurately represent the ideal !p.

Even if the !impl takes into account updating the visit counts, disallowing further
expansion beyond !tn can still be problematic:
The terminal !pred may be wrong, and the !g may actually continue beyond the assumed !tn.
In this case there are no search !ns available for the next move.
While the original !mz !arch has no issue with this, this paper outlines possible future
work where this makes a difference.

A better alternative is to force the !r and !v of each !tn child to zero.
The child relationship is considered transitive in this context, meaning that
grandchildren and all reachable child !ns have these values set to zero.

This approach also generalizes to arbitrary search depths in the sense that once a !ts is
predicted, all child !ns beyond the first !tn are set to zero !r and !v.
This is all achieved without the !nets having to learn anything about beyond-!tss.
