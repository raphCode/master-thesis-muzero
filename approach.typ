= Approach
/*
Hier beschreibst du, was du in der Arbeit neues gemacht hast, und wie du es implementiert
hast.
*/

#import "thesis.typ": citet, blockquote, neq
#import "drawings/muzero.typ": rep, dyn, pred

This chapter is divided into four parts.
I begin by reviewing the limitations of !mz and the reasons behind them, and then move on
to a discussion of how I propose to extend the !arch to handle more general !gs.
The third part proposes additional modifications that aim to improve the performance of
!mz in some cases.
The final part gives an overview about the !mz !impl written as part of this thesis,
detailing the processes of selfplay and training with my modifications.

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
<sec-limits_turn_order>

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

As outlined in @sec-muzero_limitations, the original !mz !impl is not applicable to !envs
with more than two !pls, stochasticity or general-sum !gs.
In this section, I propose modifications to the !mz !arch which lift these three
restrictions, thus improving the generality of the !algo.

The modified !algo retains compatibility with the original !mz !envs (board !gs and the
Atari suite) as special cases.
The !gs are still required to be with !pinf.

Planning ahead in a !g with more than one !pl requires some !i or assumptions about the
behavior of other !pls.
In a !2p !zsum !g, the behavior of the opponent !pl is easy to model:
He will always try to minimize the score of the other !pl.
This assumption does not hold for !mp general-sum !gs with arbitrary payoffs.

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
As outlined in @sec-limits_turn_order, in the original !mz !impl, the turn order is
hardcoded for !sp and !2p !gs and therefore represents domain knowledge about the !env.

To achieve a more general !algo, my !impl does not make any assumptions about the turn
order.
Instead, the next !pl at turn is learnt by the !dnet #dyn.
I chose this design since in !mp !gs, the next !pl at turn may depend on the !h of !as.

I propose to add an additional output head $w$ to the !dnet #dyn:
$ (s^n, r^n, w^n) = #dyn (s^(n-1), a^(n-1)) $
The output $w^n$ predicts a !prob distribution over the set of possible !pls $W$, estimating
how likely !pl $y in W$ is at turn in !s $s^n$:
$ w^n (y) = Pr(y|s^n) $

During MCTS, for each !s $s^n$ encountered, the current !pl $y^n$ is assumed to be the one
with the highest predicted !prob:
$ y^n = op("argmax", limits: #true)_(y in W) w^n (y) $

The turn output $w$ is trained like the !r $arrow(r)$, based on ground-truth labels $w_t$
given by the !g simulator during selfplay.
For this purpose an additional loss term is introduced
$ ell^w (w_(t+n), w_t^n) $
which aligns the !net !preds $w_t^n$ with their respective targets $w_(t+n)$ for all
$n = 1...K$, where $K$ is the unroll length.

In my !impl, I use a categorical cross-entropy loss for $ell^w$.

@fig-raphzero_training shows this change by the added $w$ in the !env transitions and
!dnet !preds, as well as the additional loss symbol $ell^w$.

=== maxn !MCTS
<sec-mod_maxn>

Following !mp !bi (@sec-bi_mp), the MCTS selection phase considers !n !vs for the !pl currently at turn
only.
Specifically, each !n !v is a vector:
$ arrow(Q)(s^n, a^n) = gamma arrow(v)^(n+1) + arrow(r)^(n+1) $
where $gamma$ is the !rl !df, as introduced in @sec-rl_return.

Let $Q_i (s, a)$ denote the $i$<no-join>-th component of this vector.

In !s $s^k$, maxn-MTCS then selects an !a $a^k$ as to maximize $Q_i (s^k, a^k)$ where
$i = y^k$, the !pl currently at turn, as outlined in @sec-mod_turn_pred.

Specifically, I use @eq-muzero_puct to select !as in the search tree, with 
$Q(s^k, a) = Q_i (s^k, a)$.
The constants $c_1$ and $c_2$ are detailed in @sec-eval.

=== Chance Events
<sec-mod_chance>

I model stochastic !envs with an explicit chance !pl.
He is at turn whenever a chance event occurs in the !g.

#[
#import "common.typ": wc

The occurrence of chance events is given by the !dnet as part of the turn order !pred $w$.
An additional special !pl #wc is added to the set of !pls $W$:
$ W' = W union {wc} $

If $y^n = #wc$, the current decision !n $s^n$ is assumed to be a chance event.
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

== Further Modifications

I spent a significant amount of time and effort trying to get !mz to train reliably and
accurately.
At many points I was not sure if low performance is caused by a software bug or bad design
choices related to the !nets and their training.
I therefore explored some ideas on how to improve the design of the !arch to help !net
convergence.
I outline some of the notable changes I implemented below.

=== Symmetric Latent Similarity Loss
<sec-mod_symm_latent_loss>

As outlined in @sec-effzero, #citet("effzero") already layed the groundwork for
improvements in !sampeff by introducing a similarity loss between !preds of #rep and #dyn.
However, I think that their adoption of the stop-gradient operation from !simsiam @simsiam
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

In !effz, I hypothesize that a decorrelation is already achieved by the training losses
$ell^r$, $ell^p$<join-right> and $ell^v$.

Intuitively, if the !latreps of #rep and #dyn were to collapse, the !r, !p and !v !preds
can not match their targets.
Subsequently, the training loss would stay high.
In order to accurately predict these quantities, #pred<join-right> and #dyn must encode
useful !i in the latent space from the !obs and !seq of !as, respectively.

Consequently, I see no risk of latent collapse in the !effz !arch, and propose to remove
the stop-gradient operation.

=== !TNs in the MCTS

#[
#import "drawings/muzero.typ": dyn

In !az, the !mcts uses a perfect simulator to determine the next !g !s for hypothetical
!as.
This simulator also indicates when the !g is over and there are no further moves to search
for @azero.
!mz replaces the perfect simulator with the !dnet #dyn, which is a learned model of the
!env @muzero.
]

However, I find it unexpected that this learned model does not include any concept of !tns:
#blockquote("muzero")[
  _MuZero_ does not give special treatment to terminal nodes and always uses the value
  predicted by the network. Inside the tree, the search can proceed past a terminal node -
  in this case the network is expected to always predict the same value. This is achieved
  by treating terminal states as absorbing states during training.
]

I hypothesize that this !sty may perform badly in !envs where !rs only occur in !tss:
If the !r !preds beyond !tns are not close to zero, these nonzero !rs get backpropagated
upwards and might bias the statistics in the search tree.
As the backpropagation involves summing over all future !rs #footnote[discounted by the
!rl discount factor $gamma$], the error may accumulate as the search progresses deeper
beyond !tns.

My approach is to predict the end of the !g with the !dnet.
During MCTS, !ns which are predicted to be terminal are not allowed to be expanded.
In this case, when the selection phase reaches a !tn, the backpropagation phase is
triggered immediatly with the !tn's predicted !v.

#[

#let wt = $w_frak(T)$

To realize the prediction of !tss, I add another special !pl to the set of possible turn
order !preds $W$:
The terminal !pl $#wt in W$ is at turn when the !env reaches a !ts.

]

== Overview of the !Impl

This section summarizes my !mz !impl and describes the process of search, selfplay and
training.

=== General
<sec-raphzero_general>

At a high level, my !impl is similar to !mz:
It performs selfplay with MCTS to generate training data, and trains the !nns on this
data.
However, !mz by #citet("muzero") is a large-scale !arch, running selfplay and training in
parallel distributed over multiple machines.
In contrast, my !impl is designed to operate on a single machine, and uses a single
process#footnote[parallelization is possible, but was never implemented].
Specifically, my !impl performs selfplay and training in alternation.

The time base in my !impl is the total number of steps $n$ performed in the !rl !env.
A step is defined as a single !s-!a transition, as introduced in @sec-finite_mdp.
For example, a !2p !g with 10 turns and a single chance event contributes
$10 * 2 + 1 = 21$ !env steps.
Selfplay and training metrics are logged with respect to this time base, as detailed in
@sec-raphzero_data_gen and @sec-raphzero_training.

The !nns and MCTS generally act on the set of possible !as $A$:
MCTS selects actions $a in A$, and the !net policy !preds $p$ are distributions over the
set of !as $A$.
Since my !impl also handles stochastic !gs, chance outcomes must also be included in $A$
The set of possible !as $A$ is thus the union of the set $A'$ of !as !pls can take and the
set of possible chance outcomes $C$:
$ A = A' union C $

For concrete values for all of the settings and hyperparameters introduced in this
section, refer to @sec-eval.

=== Data Generation 
<sec-raphzero_data_gen>

To generate training data, !gs are played and their !traj is recorded.
Specifically, $T$ steps are taken in each !g, until the !g terminates naturally, or the
configurable setting $M$ is reached, at which the !g is truncated.

#[
#import "common.typ": wc, wt

At chance events, that is when $w_t = wc$, the !g simulator provides the chance outcomes
$c_t$ as a distribution over !as:
#neq[
$ c_t (a|s_t) = P(s, a) $
<eq-chance_outcomes>]
where $a in C$, the set of possible chance actions.

#let tup(policy) = $(s_t, a_t, r_(t+1), w_(t+1), policy, G_t)$
At each time step $t = u, u + 1, ..., T - 1, T$ a tuple $D$ of training data
is recorded:
$ D = cases(
  tup(pi_t) &"if" w_t eq.not wc \
  tup(c_t) &"if" w_t = wc \
  ) $
The recorded !traj begins at the first time step $u$ where the !g provides an !obs, that is,
the current !pl at turn $w_t$ is not the chance !pl:
$ u = min_t t " subject to " w_t in.not {wt, wc} and 0 <= t <= T $

The tuple of training data $D$ contains the following data:
- $s_t$: !g !s
- $a_t$: !a taken
- $r_(t+1)$: experienced !r
- $w_(t+1)$: !pl at turn in the next !g !s $s_(t+1)$
- $pi_t$: MCTS !p according to @eq-mcts_policy 
- $c_t$: chance outcomes according to @eq-chance_outcomes
- $G_t$: n-step !ret according to @eq-rl_nstep

In all !gs, if the !s $s_t$ is a chance event, the !a $a_t$ is sampled from the
distribution of chance outcomes $c_t$, provided by the !g simulator:
$ a_t tilde.op c_t "if" w_t = wc $

When a !g finishes, its metrics such as the score (cumultative !r) are logged.
The metrics are associated with the total step number $n$ (see @sec-raphzero_general) when
the !g was started.
As an example, if all !gs take 10 steps, the score of the first !g is logged at $n=0$, the
second !g's score at $n=10$, and so on.

==== Warmup with random play

If the number of total !env steps $n$ is below a configurable threshold $R$ at the
beginning of a !g, !pl !as are selected randomly in the !g.
Specifically, all !as $a_t$ at time $t$ where $w_t eq.not wc$ are sampled from a uniform
distribution over the set of legal !as $A(s_t)$.
MCTS is not used in these !gs, and no !nn inferences take place.
]

This period of random play quickly generates training data so that the !dnet #dyn can
learn a model of the !env in a minimal amount of wall clock time.

==== Selfplay and Search

As soon as the number of total !env steps $n$ is above the threshold $R$, selfplay with
MCTS is used to generate data.

Dirichlet noise is blended into the root !n according to @eq-dirichlet_exploration.
!As are selected as outlined in @sec-mod_maxn.

=== Training
<sec-raphzero_training>

The latest $N$ tuples of training data are stored in a buffer for training.
The buffer additionally keeps track of the total number $i$ of training data tuples added
to the buffer, and the total number $o$ of training data tuples consumed by the training
process.
After each played !g, the generated training data is added to the buffer, and $i$ is
incremented by length of the !traj:
$i' = i + (T - u + 1)$, where $u$ and $T$ denote the first and last time step of the !g
!traj, as outlined in @sec-raphzero_data_gen.

#[
#import "common.typ": series_p, series_a, series_r, series_g, wt, wc

When data is sampled from the buffer for training, the number $o$ is incremented by $K *
B$, where $K$ is the unroll length, and $B$ the batch size:
$o' = o + (B * K)$

The ratio $o/i$ is roughly held constant at the setting $E$ by training for an appropriate
amount of times after each !g:
#neq[
$ o/i approx E $
<eq-train_ratio>]

A batch of training data is sampled from the buffer such as the first !g !s $s_0$ contains
an !obs, that is $w_0 in.not {wt, wc}$.
Training and losses are described in in @sec-muzero_atari and @sec-mod_turn_pred.
For the latent similarity loss $ell^l$ I use negative cosine similarity.

I a discrete scalar support $F$ for the !r and !v !preds, as described in
@sec-muzero_atari, but without the transform $e(x)$.

]
