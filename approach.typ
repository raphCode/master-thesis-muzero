= Approach
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
<sec-muzero_limitations>

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
From a !gtic standpoint, this requires the !g to be with !pr and of !pinf (See @sec-pr and
@sec-pinf, respectively).
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
<sec-limits_zsum>

In the !2p setting, !mz assumes that the !g is !zsum.
This assumption is built into the !arch itself, because it performs negamax search (see
@sec-negamax) during MCTS @muzero.

Negamax exploits the !zsum property by using only a single scalar for a !n's !v.
The design of the !nns (#dyn<join-right> and #pred) in !mz follows this choice and also
only predict a single scalar for !s !vs and transition !rs.

Put differently, the original !mz !impl can only learn to play in favor of one !pl, for
example white in chess or Go.
Generating strong moves for the other !pl, black in example, is achieved by negating the
!v and !r !preds for moves of this !pl.
@muzero

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

The original !mz !impl uses knowledge of the turn order during !mcts:
When !mz is configured to operate in the !2p setting, the MCTS !impl negates !n !vs at
every odd#footnote[or even, depending on who is at turn at the root !n] level in the
search tree (also known as negamax search, see @sec-negamax) @muzero.
When operating in !sp !envs, this negation is disabled and all !n !v are maximized.
The type of !env is specified in the configuration of the !algo and can thus be considered
as domain knowledge regarding the order of turns.

As outlined in @sec-limits_zsum, !mz in fact only learns a !p for one !pl in !zsum !gs.
In other words, to generate strong play for both sides, the MCTS !impl is explicitly
programmed to exploit the !zsum property and alternating turn order of the !g.

== Extension to !MP, Stochastic and General Sum !Gs

I propose an extension of !mz to more general !gs than the original !impl is capable of.
This includes !gs with chance events, more than two !pls, and therefore arbitrary payoffs.
The modified !algo retains compatibility with the original !mz !envs (board !gs and the
Atari suite) as special cases.
The !gs are still required to be with !pinf.

Planning ahead in a !g with more than one !pl requires some !i or assumptions about the
behavior of other !pls.
In a !2p !zsum !g, the behavior of the opponent !pl is easy to model:
He will always try to minimize the score of the other !pl.
This assumption does not hold for general-sum !gs with arbitrary payoffs.

My extension of !mz to !mp !gs is inspired from the !mp !bi in !gt, as introduced in
@sec-bi_mp.
My reasoning is that the subgame perfection of !bi solutions enables the RL !ag to learn a
behavior that exhibits strong play, regardless of the !as of other !pls.
Specifically, when the other !pls act optimally, the overall play should be close to the
!gtic optimal solution.
Even if the other !pls do not perform optimally, the !ag should still be able to play
reasonably.
This is especially important in !coll !gs, as it allows the !ag to compensate for bad
teammates.

I perform a number of changes to !mz.
The updated training setup with all modifications is visualized in @fig-raphzero_training.

=== Per-!PL !Vs
<sec-mod_per_player_preds>

#[
#let vector(x) = $arrow(#x) = [#x _1, #x _2, ..., #x _n] in RR^n$

!Mp !bi requires to keep track of the individual !exuts and !rs for each !pl.
I follow the design of !mp !az @mp_azero (see @sec-mp_azero) and replace all scalars
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

Note that in a !coll !g, all individual !rs are shared, as outlined in @sec-gt_collab:
$ r_i = r "for" 1 <= i <= n $

This modification is reflected in @fig-raphzero_training by adding vector arrows
$arrow(x)$ to !mp data.

=== Turn Order !Pred
<sec-mod_turn_pred>

Making informed decisions within the search tree requires not only individual !rs and !vs,
but also an understanding who can make a decision at a particular !n.
//In !mz, an alternating turn order is hardcoded in the search for !vp !gs.
In !mz, the turn order is hardcoded for !sp and !2p !gs and therefore represents domain
knowledge about the !env.

To achieve a more general !algo, my !impl does not make any assumptions about the turn
order.
Instead, the next !pl at turn is learnt by the !dnet #dyn.
I chose this design since in !mp !gs, the next !pl at turn may depend on the !h of !as.

I added an additional output head $w$ to the !dnet #dyn:
$ (s^n, r^n, w^n) = #dyn (s^(n-1), a^(n-1)) $
which predicts the !pl $w^n$ at turn in !s
$s^n$.
It is implemented as a categorical distribution $T$ over the set of !pls $W$:
$ T(s, w) = Pr(w|s) $

During MCTS, for each !s $s^n$ encountered, the current !pl $w^n$ is assumed to be the one
with the highest predicted !prob:
$ w^n = limits("argmax")_(w in W) ( T(s^n, w) ) $

The turn output $w$ is trained like the !r $arrow(r)$, based on ground-truth labels $w_t$
given by the !g simulator during selfplay.
For this purpose an additional loss term is introduced
$ ell^w (w_(t+n), w_t^n) $
which aligns the !net !preds $w_t^n$ with their respective targets $w_(t+n)$ for all
$n = 1...K$, where $K$ is the unroll length.

@fig-raphzero_training shows this change by the added $w$ in the !env transitions and
!dnet !preds, as well as the additional loss symbol $ell^w$.

=== maxn !MCTS
<sec-mod_maxn>

Following !mp !bi (@sec-bi_mp), the MCTS selection phase considers !n !vs for the !pl currently at turn
only.
Specifically, each !n !v is a vector:
$ arrow(Q)(s^n, a^n) = gamma arrow(v)^(n+1) + arrow(r)^(n+1) $
Let $Q_i (s, a)$ denote the $i$<no-join>-th component of this vector.

In !s $s^k$, maxn-MTCS then selects an !a $a^k$ as to maximize $Q_i (s^k, a^k)$ where $i =
w^k$, the !pl currently at turn, as outlined in @sec-mod_turn_pred:
$ a^k = limits("argmax")_a ( arrow(Q)_w_i (s, a) + u(s, a) ) $
where $u(s, a)$ represents some bonus term to incorporate exploration and the prior !probs
$P(s^k, a^k)$ into the decision.

=== Chance Events
<sec-mod_chance>

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

Chance events are represented by a dice in @fig-raphzero_training
#footnote[ignore the fact that the !g !ttt actually has no chance events].
Note that a chance event $s_t$ differs from regular !g !ss in the figure in two aspects:
- the !p target for training #pred are the chance outcomes $c_t$
- the target for the turn order !preds $w$ is the constant chance !pl #wc

#box[
=== Training Setup Illustration

The training setup with my proposed modifications is summarized in @fig-raphzero_training:

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
    loss_label_dynamics: $ell^r, ell^w$,
    use_vectors: true,
    value_target: "return",
    chance_event: chance_state,
  ),
  caption: [Training setup of my !impl of !mz for stochastic multi-agent !envs]
) <fig-raphzero_training>

]
]

== Further Enhancements

Furthermore, I implemented the following modifications:

=== Symmetric Latent Similarity Loss

As outlined in @sec-effzero, #citet("effzero") already layed the groundwork for
improvements in !sampeff by introducing a similarity loss between !preds of #rep and #dyn.
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

=== !TNs in the MCTS

#box(stroke: red+3pt, inset:5pt, radius:10%)[
TODO: Reason for !tns does not feel properly justified yet.\
Chapter is thus WIP.\
Maybe a better reasoning is that !nns tend to go "haywire" when used on input data never
encountered during training.
This might lead to !preds which are very off for !ns after !tns, biasing statistics of the
whole tree.
Treating !tns as "absorbing" during training means every !a must be trained, which is not
very elegant and may not generalize.
]

In !az, the !mcts uses a perfect simulator to determine the next !g !s for hypothetical
!as.
Naturally, this simulator also indicates when the !g is over and there are no further
moves to search for @azero.
These !tss are an important concept in !RL, since their value is by definition zero:
no future !rs can occur @sutton.

Tss play an important role in the case of $n$<no-join>-step !rets:
!Rets are exact for the last $n$ steps of a training !traj, because only !env !rs are
summed.
A training !traj ending at time $T$ thus has ground-thruth targets for the !vs
$v^(T-n), v^(T-n+1), ..., v^T$.
For previous time steps $t < T-n$, the !rets are bootstrapped with the !ag's !v !fn
$v_(t+n)$.
The accuracy of the training targets thus depends on the !vs estimated by the !ag.

In !mz, the n-step !ret is bootstrapped with the !v of the root !n from the tree search.
It is therefore desireable if the !v in the search tree are accurate for timesteps near
!ts.
!N !vs in a !mc search tree depend on !vs backpropagated from child !ns.
This recursive condition repeats until a !tn is reached, where the !env ultimately
provides an outcome.
In the case where !rs are sparse (zero most of the time, nonzero in !ts), !tns represent
ground-thruth anchors for !vs in the search because their !v is taken from the !env.

!mz replaces the perfect simulator with the !dnet, which provides !r !preds for
transitions between !ns.
I finde it unexpected that !mz does not include any concept of !tns:

#blockquote("muzero")[
  _MuZero_ does not give special treatment to terminal nodes and always uses the value
  predicted by the network. Inside the tree, the search can proceed past a terminal node -
  in this case the network is expected to always predict the same value. This is achieved
  by treating terminal states as absorbing states during training.
]

#[

#let wt = $w_frak(T)$

My approach is to enhance the turn order !pred $w$ with another special !pl, the terminal !pl #wt.
He is at turn when the !g (!env) reached a !ts.
This allows the MCTS to not expand !ns beyond !tss.

]
