= Approach
<approach>
/*
Hier beschreibst du, was du in der Arbeit neues gemacht hast, und wie du es implementiert
hast.
*/

#import "thesis.typ": citet, blockquote
#import "drawings/muzero.typ": rep, dyn, pred

- MuZero implementation
  - mcts behavior customizeable
    - node selection
    - policy calculation
    - action selection
  - supports arbitrary games
  - pytorch
  - typed codebase, passes mypy
- extensions / variants:
  - efficient zero
  - !tss
  - chance !ss
  - multiplayer support: teams, prediction of current player
- interplay of mcts policy and selection function
  - unstable behavior of original setup: UCT scores and visit count policy
- application to carchess / !mp !g

The original !mz !impl is limited to a specific class of !gs.
I begin by reviewing the limitations of !mz and the reasons behind them, and then move on
to a discussion of how I extend the !arch to more general !gs.

== !mz Limitations
<sec_muzero_limitations>

The !impl of !mz is designed for single-!ag !envs and !2p !zsum !gs.
Furthermore, all !gs are expected to be deterministic and of !pinf.
The causes of these limitations and some implications are briefly discussed below.

=== Determinism

In order to plan ahead, the future !s of the !env must be predictable for an intial !s $s$
and given !seq of !as.
The !dnet #dyn in !mz is a deterministic !fn and no chance !ss are modeled in the !arch.
Perhaps unexpectedly, #citet("stochastic_muzero") show that !mz's performance falls short
in a stochastic !env compared to other methods that model stochasticity.

=== !PINF

Accurately planning ahead also relies on unambiguously identifying the initial state $s$.
From a !gtic standpoint, this requires the !g to be with !pr and of !pinf (See @sec_pr and
@sec_pinf, respectively).
In the context of !rl, this means that an !obs must uniquely identify the current !s of
the !env.

This aspect best illustrated by example of the fifty moves rule in chess:
If 50 moves pass without a capture or a pawn moved, the !g may end in a draw.
While a human can deduce a history of moves from successive board !ss, the !mz !ag starts
each move afresh, given only the current !obs.
The current board !s is therefore not enough !i to distinguish a regular !g situation from
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
This is not a limitation in practice, since possible special turn order mechanics can be
modeled with the set of available !as $A$.

As an contrived example, consider castling in chess:
It may be viewed as two successive turns of the same !pl, moving king and rook separately.
However, by expanding the !a set $A$ with a castling move, the assumption about
alternating turn order still holds.

The alternating turn order is exploited by the negamax search !impl.
For single-!ag !envs, negamax is disabled altogether.
@muzero

#let sscl = [SSCL]

=== Symmetric #sscl

As outlined in @rw_effzero, #citet("effzero") already layed the groundwork for remarkable
improvements in !sampeff by introducing the #sscl.
However, their adoption of the stop-gradient operation from !simsiam @simsiam may have
been short-sighted:

In !simsiam, a Siamese !net is used to learn !latreps from images in a self-supervised
manner.
The !arch consists of two identical subnetworks that take an original and an augmented
input image, respectively, and produce a !latrep each.
During training, a distance metric is used to compare these !latreps for the same input
image.
By minimizing the distance between these !reprs, the !net is encouraged to learn a robust
and discriminative latent space that encodes useful features in the images.
However, without careful design, the model can collapse into trivial solutions where all
learned !reprs end up being very similar or even identical.
The main idea of !simsiam is that a stop-gradient operation effectively solves this
problem.
@simsiam

However, !effz operates in an !rl setting rather than a self-supervised one.
This means that the !env provides explicit training data in the form of !obs and !rs.
This data is used to train the !nets in a supervised manner, requiring them to generate
diverse !ps, !vs, and !rs from a given !obs and !seq of !as.
The exchange of !i between the different !!nets is facilitated by the !latreps.
If the !latreps were to collapse, it would become impossible to predict meaningful or
distinct output values.

This insight is consistent with the principle of contrastive self-supervised learning,
where additional negative samples are used to prevent the collapse of !latreps:
Distinct input images are fed through the two subnetworks, while the training objective is
to increase the divergence between their corresponding !latreps.
@simclr

It has been shown that it is possible to learn significant !latreps without the use of
negative samples or the stop-gradient mechanism:
#citet("ss_decorr") demonstrate that introducing a decorrelation in the latent space can
effectively steer the !net towards a non-collapsed !latrep.
In !mz, demanding different outputs from the !pnet and !dnet (!v/!p and !r, respectively)
for different !latreps can be seen as the decorrelation mechanism.

Consequently, there is no risk of !latrep collapse in the !mz !arch, making the inclusion
of a stop-gradient operation unnecessary.


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

// literal quote from paper ok?

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
