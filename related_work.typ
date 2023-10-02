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
<sec_mp_azero>

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

They evaluate their work on !mp versions of Connect 4 and !ttt:
The !nets learn to encode knowledge of the !g into search, indicating that the proposed
!mp strategy works in principle.
Performance-wise the !algo places itself below human experts.

== !smz

#[

#import "drawings/muzero.typ": rep, dyn, pred, training
#import "drawings/afterstates.typ": afterstates

#citet("stochastic_muzero") extend !mz to stochastic, but observable !envs.
The original !mz !algo is limited to deterministic !envs, due to the deterministic !preds
of the !dnet #dyn.

!smz makes use of !afs when modeling the !env.
These !afs occur after each !a of the !ag and represent an hypothetical !s of the !env
before it has transitioned to a true !s @sutton.
The idea is visualized in @fig_afterstates.
In stochastic !envs, !afs therefore can be viewed as a !s of uncertainty from which a
definitive outcome will emerge.
From a !gtic viewpoint, !afs may represent decision points of chance events.

Note that in this section, the superscript in $s_t^i$ has a special meaning in contrast to
the rest of the thesis:
It is used to differentiate distinct stochastic outcomes the !env may transition to, from
the same !s-!a pair.

#figure(
  afterstates,
  caption: [Afterstates in !smz]
) <fig_afterstates>

The transition from the !af $a s_t$ to the true !s $s_(t+1)^i$ is modeled using a chance
outcome $c_t^i$ from a finite set of $C$<join-right> possible chance outcomes.
This allows to use a deterministic model $cal(M)$ for !env transitions:
$cal(M)$ receives a !s $s_t$, the !a taken $a_t$ and a chance outcome $c_t^i$:
$ (s_(t+1), r_(t+1)) = cal(M) (s_t, a_t, c_t^i) $
This way, the task of learning stochastic transitions can be reduced to learning !afs and
distributions over the chance outcomes.

#[

#let ad = $phi$
#let ap = $psi$
#let adist = $sigma$

#let afd = [!af dynamics #ad]
#let afp = [!af !pred #ap]

In pratice, !smz introduces two new !nns, the #afd and #afp.
They are comparable to the normal dynamics and !pnet, respectively.

Specifically, the #afd predicts an !af $a s_t$ for a given !s-!a pair $s_t, a_t$:
$ a s_t = ad (s_t, a_t) $
The #afp outputs a !v $Q_t$ for an !af $a s_t$ and a !prob distribution
$adist_t = Pr(c_t^i|a s_t)$ over the chance outcomes:
$ (Q_t, adist_t) = ap (a s_t) $

The regular !dnet #dyn takes the role of predicting the transition from an !af $a s_t$ to
a true !s $s_(k+1)$ under chance outcome $c_t^i$:
$ s_(t+1), r_(t+1) = dyn (a s_t, c_t^i) $

The inference of !env dynamics from a !s-!a pair $s_t, a_t$ thus becomes a multi-step
process:
First, an !af $a s_t$ is predicted by $ad (s_t, a_t)$.
Then, the distribution over chance outcomes is obtained: $adist_t = ap (a s_t)$.
Finally, a chance outcome is sampled $c_t^i tilde.op adist_t$ and the next !s $s_(t+1)$ is
obtained using $s_(t+1) = dyn (a s_t, c_t^i)$.

!smz employs a variant of a Vector Quantised Variational AutoEncoder (VQ-VAE) @vq_vae to
learn the chance transitions and their !probs $adist$ through interactions of the !env.
VQ-VAEs have a fixed codebook size, set by a hyperparameter, which limits the number of
possible outputs.
In !smz, it is set to the maximum number of distinct chance outcomes in the !g or higher.

]

Stochastic MuZero matches the performance of !mz in deterministic !envs.
It outperforms previous approaches in stochastic domains, such as the !g 2048 and
backgammon.

]

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
  training(
    draw_latent_loss: true,
    draw_pnet: false,
    dynamics_env: n => (),
    dynamics_net: n => (),
  ),
  caption: [
    The latent loss introduced in !effz, indicated by the thick arrows.
    The other !mz losses are omitted for clarity.
  ]
) <fig_effzero_loss>

The authors employ a stop-gradient operation on the side of $s_(t+n)^0$, meaning that
gradients from the similarity loss are not applied to the !rnet #rep.
This is due to the fact that they closely modeled their !arch after !simsiam @simsiam, a
self-supervised framework that learns !latreps for images.
The authors further justify this decision by treating $s_t^n$ as the more accurate
representation and therefore using it as a target for the !dnet's !preds.
In @fig_effzero_loss, this is reflected by the unidirectional thick loss arrows.

]
