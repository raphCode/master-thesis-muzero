= Theory
<theory>
/*
Ich finde ein extra Theorie Kapitel gut, in dem man die theoretischen Grundlagend
zusammenfasst, die nötig sind um die verwendeten Algorithmen zu verstehen. 
In Research-papern ist hierfür meist kein Platz aber in einer Abschlussarbeit kann man
damit gut zeigen, dass man das Material verstanden hat.
*/

#import "thesis.typ": citet

== !RL
<sec_rl>

!Rl refers to a subset of machine learning where a decision maker learns by trial and
error while interacting with its !env.  
The decision maker takes !as that affect the !s of their !env, receives !r or penalties
based on how it performs, and updates its behavior based on this feedback.
The goal is to learn a behavior that leads to positive outcomes.
@sutton

In !rl, the entity responsible for learning and making decisions is called the !ag.
This !ag exists in an !env which consists of everything outside the agent.
Both interact continuously:
The !ag chooses !as, and the !env responds by presenting new situations to the !ag.
After each !a, the !env provides a !r - a numerical value that the !ag tries to maximize
in the long term through its action choices.
@sutton

This concept can be mathematically modeled with !mdps.

=== Finite !MDPs

Specifically, an !ag interacts with its !env in discrete time steps $t = 0, 1, 2, ...$.
At each time $t$, the !ag receives an !obs of the current !s $s_t in S$ and then chooses
an !a $a_(t+1) in A(s)$ to take.
The !a transitions the !env to a new !s $s_(t+1)$ and generates a numerical !r $r_(t+1) in
R subset RR$ for the !ag.
$S$<no-join>, $A$<join-right> and $R$ denote the set of all !ss, !as and !rs respectively.
@sutton

Note that the subscript used to represent an !a that occurred between $s_t$ and $s_(t+1)$
varies in different literature:
Some associate this !a with time step $t$ @sutton, others with $t+1$ @muzero.
In this thesis, I will follow the first convention given in #citet("sutton").
Further confusion may arise when comparing this to the !impl that accompanies this thesis:
I store tuples $(s_t, a_t, r_(t+1))$ together at the same array index because this is most
convenient for training.


The concept of !seql decision making can be formalized with finite !mdps.
Finite means the sets of possible !ss $S$, !as $A$ and !rs $R$ each have a finite number
of elements.
Beside these sets, a !mdp is characterized by the dynamics !fn $p : S times R times S
times A -> [0, 1]$:
$ p(s', r|s, a) eq.def Pr{s_(t+1)=s', r_(t+1)=r|s_t=s, a_t=a} $
for all $s', s in S, r in R, a in A(s)$.

It describes the !prob of the !env ending up in !s $s'$ and yielding !r $r$ when executing
!a $a$ in !s $s$.
@sutton

=== !Epis and !Rets

In some scenarios the interaction between !ag and !env naturally ends at some point, that
is, there exists a final time step $t_n$.
In !gs this happens when the match is over.
In !rl such a series of interactions is called an !epi.
@sutton

The !seq of visited !ss $s_0, s_1, s_2, ..., s_n$ is referred to as a !traj.
Depending on the context, a !traj may also include the !as and !rs associated with the
transitions between !ss.
The last !s $s_n$ is also called the !ts. 
@sutton

To translate multiple !rs earned over a period of time into a singular value that guides
the agent in making optimal decisions, we use the concept of a !ret.
The !ret is a specific !fn of the !r !seq.
The !ag's learning objective is then defined as maximizing the !exret.
In a simple case, the !ret $G_t$ may be defined as the sum of all !rs occurring after time
step $t$:
$ G_t eq.def r_(t+1) + r_(t+2) + r_(t+3) + ... + r_n $

Another approach is to use a discounted !r.
The intuition is to value !rs far in the future less than immediate !rs.
For this purpose, a hyperparameter $gamma$ is introduced, called the !df.
The !ret $G_t$ is then calculated as
$ G_t eq.def r_(t+1) + gamma r_(t+2) + gamma^2 r_(t+3) + ... =
sum_(k=0)^(n-1) gamma^k r_(t+k+1) $
where $0 <= gamma <= 1$.

The !df affects how valuable future !rs appear in the present moment:
For example, a !r that arrives k time steps in the future will have its current value
reduced by a factor of $gamma^(k-1)$ compared to if it had arrived immediately.
@sutton

=== !Ps and !V !Fns

A !p is a formal description of the !ag's behavior.
Given a !s $s_t$, the !p $pi(a|s)$ denotes the !prob that the !ag chooses !a $a_t=a$ if
$s_t=s$.
@sutton

Since the !p makes statements about the future behavior of the !ag, one can now define the
!exret.
The !exret describes the expected value of the !ret $G_t$ in !s $s_t$, if the !p $pi$ is
followed.
It is therefore also called the !v $v_pi(s)$ and defined as
$ v_pi(s) eq.def EE_pi [G_t|s_t=s] = EE_pi [sum_(k=0)^(n-1) gamma^k r_(t+k+1)
#move(dy: -3pt, scale(y: 300%, [$|$<no-join>])) s_t=s] $
for a !s $s_t$ when following !p $pi$.
@sutton

The !v is thus an estimate of how good it is for the !ag to be in a particular !s,
measured by the objective !fn, the !ret.
@sutton

In !tss, the !v is always zero per definition:
There are no future !rs possible.
@sutton

== !GT

This section introduces basic !gtic concepts, just enough to provide justification for the
design and behavior of !mz.
It also provides foundations to discuss the limitations of the original !mz !impl in
TODO: label.

!Gt is a broad interdisciplinary field that uses mathematical models to analyze how
individuals make decisions in interactive situations.
It is commonly used in economics, but has other widespread applications.
@gtheo

In this thesis I am interested in !gtic analysis of the behavior and interaction of
multiple !pls in a !g.
!Gtic considerations can be used to find optimal behavior in a given !g under some
assumptions.
In this way, !gt can provide a foundation for what should or even can be learned by a !rl
system.
@gtheo

=== Basics

I begin with introducing some terminology and basic concepts:

A !g in !gt is a very general concept and applies to more than what the common usage of
the word "!g" refers to.
A !g may denote any process consisting of a set of !pls and a number of !dps.
Additionally, the !i and !as available to each !pl at each !dp must be defined.
@gtheo

When the !g is played, each !pl has to make a choice at their !dps.
This choice is often referred to as a move or !a.
A play of a !g consists of the moves made by each !pl, so the !dps are replaced by
concrete choices.
Such a !seq of moves is also called a !h.
The !h is also defined for any situation in the middle of the !g.
Since it contains all the moves made so far, it identifies the current !s of the !g.
@gtheo

A !g also needs to have a definition of its possible final outcomes.
These are defined in terms of a numerical payoff, one for each !pl.
Payoffs for multiple !pls are noted as a tuple, also called a payoff vector in this case.
@gtheo

The !gs considered in thesis are finite, meaning that the number of possible choices is
limited.
Finiteness also applies to the length of the !g, it should end at some point.
This may sound trivial since most !gs encountered in the real world are finite.
However, !gt is not limited to finite !gs.
It is therefore important to note that some of the !gt statements in this chapter only
apply to finite !gs.
@gtheo

!Gs that involve randomness, such as throwing a dice, are said to have chance events.
These chance events can be seen as !dps where a special !pl, the chance !pl, has his turn.
A dice throw can therefore be modelled with a !dp of the chance !pl with 6 possible !as,
all with equal !probs.
@gtheo

A behavior is described by a !sty:
It can be seen as a complete plan which describes which move to take at any given !dp of
the !g.
@gtheo

Given a !sty for each !pl, one can calculate the !probs of future moves and !g outcomes.
For any !pl and any !s of the !g, it is thus possible to derive the expected payoff, also called !exut.
@gtheo

Both !gt and !rl are about decision making, so naturally they employ similar concepts.
However, comparable concepts have slightly different terminology in the two fields.
@tbl_gt_rl_terms shows a tabular overview of related terms between !gt and !rl.

#figure(
  table(
    columns: (auto, auto),
    [*!Gt*], [*RL*],
    [!g], [!env],
    [!pl], [!ag],
    [decision / move / !a], [!a],
    [!sty], [!p],
    [outcome], [!ts],
    [payoff], [terminal !r],
    [!exut], [!exret],
    [!h], [!traj],
  ),
  caption: [Comparison of similar concepts in !gt and !rl],
) <tbl_gt_rl_terms>

=== Properties of !Gs

!Gt may differentiate !gs according to different properties which are explained in the
next chapters.

==== Payoff Sum

!Zsum !gs are !gs in which the sum of all !pls' payoffs equals zero for every outcome of
the !g.
It is a special case of the more general concept of constant-sum !gs, where all payoffs
sum to a constant value.
In other words, a !pl may benefit only at the expense of other !pls.
Examples of !2p !zsum !gs are tic-tac-toe, chess and Go, since only one !pl may win
(payoff of 1), while the other looses (payoff of -1).
Also poker is considered a !zsum !g, because a !pl wins exactly the amount which his
opponents lose.

Contrastingly, in non-!zsum !gs, no restriction is imposed on the payoff vectors.
This situation arises when modeling real-world economic situations, as there may be
gains from trades.

==== !I

An important distinction are !gs of !pinf and !impinf.
A !pinf !g is one in which every !pl, at every !dp in the !g, has full knowledge of the !h
of the !g and the previous moves of all other !pls.
A !pl is thus fully and unambiguously informed about the current !s of the !g when he is
at turn.
Chess and Go are examples of !gs with !pinf.
This is because each !pl can see all the pieces on the board at all times.
@gtheo

!Gs with randomness can also have !pinf if the outcome of all random events is visible to
the next !pl at his turn.
An example is backgammon: when a !pl needs to make a decision, he has !pinf about what
number the dice rolled.

Conversely, a !g with !impinf leaves !pls in the dark about the exact !s of the !g.
!Pls having to make a decision do not observe enough !i to distinguish between several
past !g !hs.
They thus have to make decisions under uncertainty.
The card !g poker is an example of an !impinf !g because the other !pls' cards are
concealed.
@gtheo

==== !PR

The concept of perfect recall describes that a !pl never forgets all his past choices and
information he got.
This allows !pls to unabigously separate past, present und future !ss.
@gtheo

!Pr is often assumed in !gt, since it is a requirement for e.g. the rationality of !pls,
which is explained in @sec_bi_mp.
@gtheo

==== Simultaneous / !Seql Moves

!Gs can be classified according to whether the !pls make their moves at the same time or
one after the other.
In the first case, the moves are said to be simultaneous, and the !g is also called a
static !g.
@gtheo2

An example of a simultaneous !g is !rps:
Both !pls choose their hand sign at the same time.
Because of the static nature of the !g, it is not possible to observe the !a chosen by the
other !pl and react to it.
@gtheo2

In fact, non-observability is the defining aspect of simultaneous !gs:
Take, for example, an election where all voters make their choices without knowing what
anyone else has chosen.
Even though the votes are not literally cast at the same time, the process is still an
example of a simultaneous !g.
@gtheo2

In contrast, in a !seql !g the !pls take turns in succession.
These !gs are also called dynamic !gs or !exf !gs.
To distinguish them from simultaneous !gs, a !pl making a decision must have !i about
the previous decisions of other !pls.
It is important to note that only some !i is required, not necessarily !pinf.
@gtheo2

==== Determinism

A !g that involves no chance events is said to be deterministic.
For instance, chess and Go are deterministic !gs since the !g !s only depends on the moves
of the !pls.
@gtheo

==== Cooperation and Collaboration

Traditional !gt divides !gs into two categories, cooperative and non-cooperative.
The cooperative approach studies !gs where the rules are only broadly defined.
In fact, the rules are kept implicit in the formal specification of the !g.
@gtheo

Since the rules are not specific enough to analyze individual decision making, cooperative
!gt looks instead at coalitions of !pls.
These coalitions assume that !pls can commit to work together through binding agreements
or through the transfer of decision-making power.
@gtheo

Overall, cooperative !gt provides a framework for understanding how different parties can
work together toward common goals.
@gtheo

Unlike the other branch, non-cooperative !gt requires an exact specification of the rules
of the !g.
For this reason, it is also known as the theory of !gs with complete rules.
It allows an analysis of the individual decisions of !pls without relying on commitment
devices.
!Pls are assumed to act solely out of self-interest, i.e. to choose !as that maximize
their payoff.
@gtheo

However, a third category can be identified:
In a collaborative !g, all !pls work together as a team, sharing the outcomes and thus
payoffs.
A team is defined by a group of !pls who have the same interests, albeit the individual
information !pls have may differ.
Since the rewards and penalties of their !as are shared, the challenge in a collaborative
!g is working together to maximize the team's payoff.
In contrast, cooperation among individuals may involve different payoffs and goals of the
different !pls.
#cite("collaborative_games", "eco_theo_teams")

=== !Exf

#[

#import "drawings/gametree.typ": draw_gametree, n, nodetree

In !gt, !gs can be modeled in different forms.
These forms provide a formal !repr of the arbitrary rules of a !g.
The !gs studied in this thesis are !seql, of !pinf and may involve multiple !pls.
This makes the !exf an appropriate choice.
It can be seen as a kind of flowchart that shows what can happen during the course of
playing the !g.
@gtheo

#let root = nodetree(
  [C],
  n([$P=1/2$<no-join> heads], [1],
    n([go], [2],
      n([left], (2, 3)),
      n([right], (4, 1)),
    ),
    n([stop], (-1, 0)),
  ),
  n([tails $P=1/2$<no-join>], [2],
    n([stop], (0, -1)),
    n([go], [1],
      n([left], (1, 2)),
      n([right], (5, 3)),
    ),
  ),
)

#figure(
  draw_gametree(root),
  caption: [!Exf example of an artificial !g with two !pls and a chance event]
) <fig_exf_example>

The !exf looks similar to a regular tree in computer science.
It is accordinly also known as a !g tree.
Each !n represents a !dp for a !pl, with the !g starting at the root !n.
Outgoing edges from a !n are labeled with !as the !pl can choose.
The leaf !ns are the outcomes of the !g and specify the payoff vectors.
In summary, it can be stated that the !exf enumerates all possible !hs of the !g in a
tree-like form.
@gtheo

An example of the !exf of an artificial !g is given in @fig_exf_example.
The root !n denotes the start of the !g, in the example it is labeled with~C.
In this case the !n is meant to represent a chance event, specifically a coin flip.
Therefore, it has two possible outcomes with equal !probs of $1/2$ each, and the edges are
labeled with _heads_ and _tails_ respectively.

The !g continues with either !pl~1 or~2, depending on the chance event.
The coin flip therefore determines the starting !pl.
The first !pl to move has two available !as, _stop_ and _go_.
The former ends the !g immediatly with a reward of~-1 for the !pl which chose _stop_, and
zero for the other one.
If the !g continues, the other !pl is given a choice to go left or right, after which the
!g ends with the payoffs in the !tns.

The !exf also allows an visual explanation of subgames.
The !exf of a subgame is a subset of the original !g's !g tree.
In the example of @fig_exf_example, if a !g were to start at any of the !ns labeled with~1
or~2 and shares all !ns below, this is a subgame of the original !g.
@gtheo

For example, @fig_exf_subgame shows two particular subgames from the !g in
@fig_exf_example:

#figure(
  stack(
    dir: ltr,
    spacing: 10mm,
    draw_gametree(root.children.at(0)),
    align(horizon, draw_gametree(root.children.at(1).children.at(1)))
  ),
  caption: [Some subgames of the !g in @fig_exf_example]
) <fig_exf_subgame>

]

=== Solutions, Equilibria and Optimal !Stys

A goal of !gt is to assign solutions to, or "solve" !gs.
A solution is a set of !stys, one for each !pl, that leads to optimal play.
The optimality is defined according to some condition or assumption.
A !g can be said to be solved when a solution can be obtained with reasonable resources,
such as computing time and memory.
#cite("gtheo", "phd_games")

In non-cooperative !gt, the goal of an optimal solution is always to maximizing the !pl's
payoff.
@gtheo

==== !BI

In the case of !pinf !gs with !seql moves, an optimal solution can be computed with a
simple !algo, called !bi.

===== !SP

!Bi is best introduced with a !sp !g, as it makes the !g analysis straightforward.
Consider for example this !g in !exf, as shown in @fig_bi_sp:

#[

#import "drawings/gametree.typ": draw_gametree, n, l, r, nodetree, get_optimal_strategy

#let root = nodetree(
  backpropagate: true, 
  [A],
  l([B], l(3), r(1)),
  r([C], l(0), r(4)),
)

#let (node1, node2) = root.children

#figure(
  draw_gametree(root),
  caption: [!Bi in a simple !sp !g]
) <fig_bi_sp>

The optimal !sty can now be computed bottom-up, starting at the leaf !ns:
If the !pl were already at~!dp #node1.content, he would certainly choose
!a~#node1.backprop_info.action since this results in the bigger payoff
of~#node1.backprop_info.utility over all alternatives.
Therefore, !n~#node1.content can be assigned an utility of~#node1.backprop_info.utility.
Likewise, the utility of !dp~#node2.content can be determined to
be~#node2.backprop_info.utility, as the !pl would always
choose~#node2.backprop_info.action.
Now that the utilities of~#node1.content and~#node2.content are found, the
optimal decision at~#root.content can be identified to be~#root.backprop_info.action with
the same reasoning.
Since~#root.content is already the root !n, the optimal !sty is thus
${ #get_optimal_strategy(root).join(" ") }$.
@gtheo

]

===== !MP
<sec_bi_mp>

In a !mp setting, the !stys of other !pls influence the course of the !g and thus the
utility of !ss.
Since it is not immediately apparent what other !pls will do, it raises the question of
whether an optimal !sty can even be determined.
However, !gt can answer this question in the affirmative.
By introducing the concept of rationality, one can make robust assumptions about the
behavior of the !pls and the !stys they will select.
@gtheo

The concept of rationality is built on the premise that !pls aim to maximize their
individual payoffs.
Importantly, rationality also includes the understanding that each !pl is aware that
others will act on this premise.
Consequently, a !pl can adjust his !sty accordingly.
However, all other !pls can also infer this change in !sty and adjust their own !stys
accordingly... ad infinitum.
@gtheo

It can be shown that this reasoning converges to a solution.
The constraint that all !pls try to maximize their payoff has an important implication:
No !pl can improve by choosing a different !sty, as long as he expects all other !pls to
adhere to the solution.
The solution is therefore self-enforcing and no !pl has incentive to deviate.
Such self-enforcing !sty combinations are known as Nash equilibria.
@gtheo

#[

#import "drawings/gametree.typ": draw_gametree, l, r, nodetree, get_optimal_strategy

#let root = nodetree(
  backpropagate: true, 
  1,
  l(2,
    l(3, l((1, 2, 2)), r((3, 5, -1))),
    r(3, l((8, 3, 4)), r((2, 6, -4))),
  ),
  r(2,
    l(3, l((3, 7, -3)), r((-3, 6, 7))),
    r(3, l((-8, 5, 2)), r((3, -4, 1))),
  ),
)

#let (node1, node2) = root.children
#assert(node1.content == node2.content)
#let node_leftmost = node1.children.at(0)

Such a solution for !mp !gs can be computed as well with !bi.
@fig_bi_mp visualizes the process for an example !g with three !pls, each having one !dp.
The !ns are now labeled with the number of the !pl at turn, so there are multiple !ns with
the same number.

#figure(
  draw_gametree(root),
  caption: [!Bi in a !mp !g with three moves]
) <fig_bi_mp>

The tree looks a bit more complicated since a !mp !g involves payoff vectors instead of
single scalar payoffs.
However, the reasoning is exactly the same as in the !sp scenario:
Starting at the last !dps in the tree, !pl~#node_leftmost.content can decide which !a to
take.
He is only interested in maximizing his payoff, so he only looks a the third entry of the
payoff vectors.
In @fig_bi_mp, this is illustrated by underlining the respective entry in the tuple of
payoffs.
In the specific case of the leftmost !dp in the tree, !pl~#node_leftmost.content has the
possible
outcomes~#node_leftmost.children.map(c => str(c.utility.at(c.parent_player - 1))).join(" and ").
As the higher payoff is~#node_leftmost.backprop_info.utility, he will always choose
!a~#node_leftmost.backprop_info.action.
Thus, the leftmost !n can be assigned the payoff vector~#repr(node_leftmost.utility) since
that is how the !g will end from this !dp onwards.

Similarly, the other !ns of !pl~#node_leftmost.content can be processed and utilities
assigned that maximize !pl~#node_leftmost.content's payoff.
Next, we can move one step upwards in the tree, looking at !pl~#node1.content's decision.
He is only interested in the payoffs relevant to him, which are in the second entry of the
payoff vectors.
Again, this is illustrated in @fig_bi_mp with underlining the corresponding entries.
Consider for instance the left !n labeled with~#node1.content:
The !pl will choose !a~#node1.backprop_info.action, since that gives him the higher payoff
of~#node1.backprop_info.utility.
In the !exf, this means the utility~#repr(node1.utility) of the respective child !n can be
propagated upwards and assigned to !pl~#node1.content's decision node.

Analogously, !pl~#root.content reasons that~#root.backprop_info.action is his best choice.
Overall, three rational !pls will choose the respective
!as~#get_optimal_strategy(root).join(", ").

===== Chance Events

#[

#import "drawings/gametree.typ": draw_gametree, n, nodetree

#let (p1, p2) = ((1, -2, 4), (-3, 0, 1))
#let (Pa, Pb) = (0.4, 0.6)

#let exp_payoff = p1.zip(p2).map(((a, b)) => a * Pa + b * Pb)

If the !g involves chance events, chance !ns may be replaced by their expected outcome
@gtheo.
For example, consider the chance !n depicted in @fig_bi_chance with the two outcomes
#repr(p1) and #repr(p2) and !probs #repr((Pa, Pb)) respectively.
The expected payoff of #repr(exp_payoff) is calculated by weighting the possible payoffs
by their !probs.


#let root = nodetree(
  exp_payoff,
  n([P = #Pa], p1),
  n([P = #Pb], p2),
)

#figure(
  draw_gametree(root),
  caption: [Expected payoff of a chance !n]
) <fig_bi_chance>

]

===== !ZSUM !Gs

For !2p !gs with !zsum payoffs, practical !impls of !bi may keep track of only a single
!pl's payoff scalar in the !g tree, for example the first !pl.
The payoff for the other !pl is implicitly given by the !zsum property.
While such an !impl navigates the !g tree normally for moves of the first !pl (maximizing
the payoff), it has to minimize the payoff for the other !pl.
Such an !impl is known as minimax search.
@ab_pruning

Instead of selecting the minimum payoff for the other !pl, the payoff scalar can also be
negated for the moves of the other !pl.
In this case, the !algo can handle all moves in the same manner by maximizing the payoff.
This variant is called negamax.
@ab_pruning

==== Subgame Perfection

An interesting property of the solutions visualized in @fig_bi_sp and @fig_bi_mp is that
they also contain optimal !as for !g !s which are not part of the overall optimal !sty.
For example, in @fig_bi_mp, the right !n of !pl~#node2.content is not part of the optimal
play.
However, if !pl~#node2.content would ever find himself in this !g !s (maybe through a
mishap of !pl~#root.content), he knows that his best option
is~#node2.backprop_info.action.

]

If an optimal solution to a !g also contains optimal solutions for all its subgames, the
solution is said to be subgame perfect. 
In the case of !bi, the computed solution is always subgame perfect.
@gtheo

Subgame perfection is an desireable property of a solution, since it allows !pls to react
to other !pl's deviations from the optimal !sty.
Compared to all !pls adhering to their optimal !stys, this means:
- in a non-cooperative !g: exploiting opponent's mistakes, to potentially achieve a higher payoff
- in a collaborative !g: compensating for mistakes of teammates

== !MCTS
<sec_mcts>

!Mcts (MCTS) is a stochastic !algo that can be applied to !seql decision problems to
discover good !as.
It takes random samples in the decision space and collects the outcomes in a search tree
that grows iteratively.
The tree thus contains statistical evidence of available !a choices.
#cite("mcts_survey", "mcts_review")

!Mcts is attractive because of its properties:
First, it can handle large !s spaces due to the random subsampling.
This is essential for !gs or problems where the decision space cannot be fully searched.
Second, MCTS is an anytime !algo:
It can be interrupted at any point and returns the best solution found so far.
This is important for e.g. !gs where only a limited time budget is available for move decisions.
Likewise, allowing more computing time generally leads to better solutions.
Lastly, MCTS can be utilized with little or no domain knowledge.
#cite("mcts_survey", "mcts_review")

!Mcts grows a tree #footnote[typically a !g tree] in an asymmetric fashion, expanding it
by one !n in each iteration.
Each !n in the tree represents a !g !s $s$ and stores a visit count $n_s$, that indicates
in how many iterations the !n was visited.
A !n also keeps track of its mean !v $v_s$ as approximated by the !mc simulations.
In the basic variant, each iteration of the !algo consists of four phases, as illustrated
in @fig_mcts_phases.

#[

#import "drawings/mcts.typ": draw_mcts

#let labels = (
  [Selection],
  [Expansion],
  [Simulation / Rollout],
  [Backpropagation],
)

#figure(
  grid(
    columns: labels.len(),
    column-gutter: 10mm,
    row-gutter: 5mm,
    ..labels.map(strong).map(align.with(center + bottom)).map(par.with(justify: false)),
    ..for i in range(labels.len()) {
      (draw_mcts(i + 1), )
    }.map(align.with(center))
  ),
  caption: [The four phases of one iteration of !mcts],
) <fig_mcts_phases>

#let phase(n) = labels.at(n - 1)

#let phase_heading(n) = [
  #strong(phase(n)):\
]

The phases are executed in the following order, unless noted otherwise:

#phase_heading(1)
The !algo descends the current tree and finds the most urgent location.
Selection always begins at the root !n and, at each level, selects the next !n based on an
!a determined by a tree !p.
The tree !p aims to strike a balance between exploration (focus areas that are not sampled
well yet) and exploitation (focus promising areas).\
This phase terminates in two conditions:
- a !ts of the !g is reached. In this case, the !algo skips to #phase(3).
- the !n corresponding to the next selected !a is not contained in the tree yet.
#cite("mcts_survey", "mcts_review")

#phase_heading(2)
Adds a new child !n to the tree at the position determined by the #phase(1) phase.
#cite("mcts_survey", "mcts_review")

#phase_heading(3)
The goal of this phase is to determine a !v sample of the last selected !n.
If the !n is already a !ts, the outcome $z$ of the !g can be used directly and the !algo
skips to #phase(4).\
In most cases however, the !n is somewhere "in the middle" of the !g.
The !algo then takes !as at random until a !ts is reached.
Subsequently the outcome $z$ of this !ts is used to proceed in the #phase(4) phase.
#cite("mcts_survey", "mcts_review")

How the !g's outcome is translated into a !n !v may depend on the specific !g and MCTS
!impl.
In the case of a !sp !g, the !v may equal the single payoff scalar @muzero.
The MCTS !impl may also perform a stochastic equivalent to !bi, so in a !mp !g, a payoff
vector with multiple entries may be required @mp_azero.
For !2p !zsum !gs, a single payoff scalar may suffice, following the idea of minimax /
negamax search #cite("azero", "ab_pruning").

Intermediate !g !ss visited during this phase are not stored or evaluated in any way.
The !algo's "!mc" property stems from the random choice of !as during this phase.
An optimisation over random !as may be to sample !as from another !p, also referred to as
the rollout !p.
#cite("mcts_survey", "mcts_review", "alphago")

#phase_heading(4)
Propagates the !v $z$ obtained in the #phase(3) upwards in the tree, updating the
statistics of all ancestor !ns.
For each !n on the path to the root, the visit count $n_s$ is incremented and the !v
statistics of the !n is updated to include $z$.
#cite("mcts_survey", "mcts_review")

For example, a MCTS !impl with scalar !vs may be interested in the average !v $v_s$ of all
simulations that passed through the !s $s$.
It can store the sum of all simulation !vs $u_s$, and use the visit count $n_s$ to
calculate the !n !v
$ v_s = cases(
  u_s / n_s & "if" n_s eq.not 0,
  0 & "else",
) $
During the #phase(4) phase, the following updates would then be performed:
$ n_s' = n_s + 1 \
  u_s' = u_s + z $

#cite("muzero", "alphago")

]

== !mz and Precursors

!mz is a state-of-the-art deep !rl !algo that builds on previous generations of closely
related !rl !algos.
By understanding its precursors, we can gain valuable insight into !mz's design and how it
evolved from traditional game playing programs.

This section explores and explains the predecessors, they are listed in ascending order of
publication date.
They build on each other and eventually lead to !mz, which is explained in detail in
@sec_muzero.

Since the first two !algos, !ago and !agoz, focused on the !g of Go, I start with a
overview of previous attempts at Computer Go.

=== Computer Go

Go is a !2p, !zsum board !g where !pls place their (black and white respectively) stones
on a 19x19 grid.
Go has long been known as a very difficult !g for computers to play.
The reason is the high complexity in the !g tree:
A complete !g tree would be very large in both height (Go !gs can span hundreds of moves)
and breadth (e.g. the empty board provides $19^2 = #(19 * 19)$ possible locations for
placing a stone).
This makes an exhaustive search computationally intractable.
#cite("computer_go", "phd_games")

Go programs preceeding !ago therefore use !mcts to subsample !ss in the !g tree.
A search !p is used to prioritize promising moves, reducing the effective breadth of the
search tree.
To estimate !vs of !ns the tree, complete rollouts are simulated until the end of the !g.
To obtain robust !v estimations, many rollouts are needed.
Obviously, the !p for selecting !as during rollouts is crucial for the determined !vs and
resulting performance.
#cite("mcts_balancing", "mcts_balancing_practice", "pachi", "fuego")

Since a high number of rollouts with many simulation steps each are needed, the !p !fns
are required to be fast.
Prior work to !ago therefore uses very simple !ps in the search tree and during rollout !a
selection.
These !p !fns can be hand-crafted heuristics based on features extracted from the Go board
@go_hand_patterns. 
Another possibility is to learn a shallow !fn based on a linear combination of board
features
#cite("go_learn_patterns", "mcts_balancing", "mcts_balancing_practice").
All features are hand-crafted and use domain-specific knowledge of the !g.
#cite("go_hand_patterns", "go_learn_patterns", "pachi", "fuego").

However, evaluating board positions via rollouts has been shown to be often inaccurate
@go_mcts_limits.
There have been efforts to include a !v !fn that directly estimates the !v of a position
without any rollouts @on_offline_uct.
But again, the !v !fn used by #citet("on_offline_uct") is simple and based on a linear
combination of hand-crafted features.
Hence, its accuracy is limited and complete rollouts are still required to achieve good
performance @on_offline_uct.

The performance of these approaches is limited, reaching only strong amateur level play
#cite("fuego", "pachi", "go_learn_patterns").

=== !ago
<sec_alphago>

!ago by #citet("alphago") is a novel and successful approach to the !g of Go with the full
19x19 board size.
Like prior work, it is based on !mcts, enhanced with a !p and !v !fn.
Unlike previous approaches, !ago uses deep !nns for these !fns.
Deep !nns can give much more accurate approximations than previously used heuristics or
shallow !fns.

Specifically, all !nets in !ago are deep convolutional !nns (CNN) that operate direcly on
a 19x19 images of the Go board.
The input to all !nets is a simple !repr of the current board:
Several layers of images encode the positions of stones and hand-crafted features on the
board in the current as well as past moves.

#[

#import "drawings/alphago.typ": training_pipeline, sl, rl, roll, v

#let slnet = [SL !p !net #sl]
#let rlnet = [RL !p !net #rl]
#let rollnet = [rollout !p !net #roll]
#let vnet = [!vnet #v]

!ago uses multiple !nns for the !mcts.
They are trained with a multi-stage pipeline that includes supervised and !rl.
@fig_alphago_train shows an overview of the training process and the !nns used in !ago.
Some of the !nns are only used to generate training data for other !nets.

#figure(
  training_pipeline,
  caption: [The !ago training pipeline]
) <fig_alphago_train>

The pipeline starts with supervised learning (SL) on the KGS Go dataset, which contains Go
games played by human experts.
A !nn trained on this data predicts which moves humans would play in a given board
situation.
This is actually not a novelty on its own, since previously CNNs have already been used
for this task #cite("go_cnn_2008", "go_cnn_2014a", "go_cnn_2014b").
However, the use of a larger convolutional !nn allowed them to reach a higher accuracy
than previous attempts.

Two !nns are trained on the KGS Go dataset, a big #slnet and a smaller one, the fast
#rollnet.
The #rollnet is less accurate, but an order of magnitude faster to evaluate than the
#slnet.

The next stage in the training pipeline uses !rl (RL) and selfplay to improve the #slnet.
The autors call the resulting !net #rlnet, it can be seen as a fine-tuned version of the
#slnet.
The idea is to train the !net towards the relevant goal of winning !gs, which does not
necessarily align with predicting expert moves perfectly @mcts_balancing.

First, the #rlnet is initialized to the same structure and weights of the #slnet.
The #rlnet is then trained with !p gradient #cite("policy_gradient", "reinforce") to win
more !gs against previous versions of the #rlnet.
For this, games are played between two agents which select !as sampled from !preds of the
#rlnet.
One agent uses the current version of the #rlnet, the other one a random older iteration.
The authors argue that randomizing from a pool of opponents prevents overfitting and
stabilizes training.

The selfplay in !ago does not use any search.
The final iteration of the #rlnet already plays Go better (without search) than the
strongest available open-source Go program.

The last stage trains a #vnet that evaluates positions directly.
It is trained in supervised manner to predict the !g outcome from the current board
position, assuming strong !pls.
A suitable dataset for this task requires a large number of Go !gs and strong play of both
!pls.
The authors used the #rlnet and selfplay to generate a new dataset that fulfills these
requirements.

Specifically, each selfplay !g is carried out in three phases:
First, a random number $U$ is sampled uniformly $U tilde.op "unif"{1, 450}$.
Then, the moves at time steps $t = 1, ..., U - 1$ are sampled from !preds of the #slnet,
$a_t tilde.op #sl (dot.c|s_t)$.
Second, a single move $a_U$ is sampled from the legal moves.
Lastly, the #rlnet is used to generate the remaining moves $t = U + 1, ..., T$ until the
!g terminates, $a_t tilde.op #rl (dot.c|s_t)$.
The !g is then scored to determine its outcome $z_T$.
From every !g, only a single training example $(s_(U+1), z_T)$ is added to the selfplay
dataset.
The final dataset contains positions from 30 million distinct !gs.

Finally, three !nets (the two !pnets #sl, #roll and the #vnet) are combined in a variant
of !mcts (MCTS).

In the MCTS selection phase, !preds from the #slnet are used.
The #slnet was found to perform better in this job than the #rlnet.
The !preds of the #slnet are combined with the !vs already present in the tree to balance
guidance from the !net !preds, exploitation and exploration.

Specifically, in the search tree, each !n corresponds to a !g !s $s$, and the outgoing
edges represent legal !as.
Each edge stores an !a !v $Q(s, a)$, the visit count $N(s, a)$ and a prior !prob $P(s, a)$
derived from !preds from the #slnet.
In each time step $t$ of the MCTS selection phase, the child !n corresponding to the !a
$a_t$ is selected from the !s $s_t$
$ a_t = limits("argmax")_a ( Q(s_t, a) + u(s_t, a)) $
to maximize the !a !v plus a bonus.

The bonus term $u(s, a)$ is initially proportional to the prior !prob but decays with
repeated visits to encourage exploration:
$ u(s, a) prop P(s, a) / (1 + N(s, a)) $

This selection strategy is a variant of the UCT !algo.
UCT (and its name) itself is derived from applying the UCB (Upper Confidence Bounds) !algo
to trees.
#cite("puct", "uct", "alphago")

The MCTS selection phase traverses the tree as usual, until time step $L$, where it
reaches a leaf !n that may be expanded (if it is not a !ts).
During expansion, the new !n corresponding to !s $s_L$ is processed by the #slnet to yield
the prior !probs
$ P(s_L, a) = #sl (a|s_L) $

During the MCTS simulation phase, the new !n is evaluated using a combination of !mc
rollouts and the #vnet.
Specifically, for the !s $s_L$, a rollout until !g end is performed using the #rollnet to
yield a !v $z_L$.
The rollout !v $z_L$ is blended with the !pred from the !vnet, $#v (s_L)$, to the overall
!v $V(s_L)$, using a mixing factor $lambda$:
$ V(s_L) = (1 - lambda) #v (s_L) + lambda z_L $

The !algo performed best with a mixing factor $lambda = 0.5$, that is, equal weighting of
the rollouts and !vnet.
However, even without any rollouts at all ($lambda = 0$<no-join>), !ago performed better
than previous computer Go programs.

One MCTS iteration is concluded by backpropagating $V(s_L)$ up in the tree.
All edges on the search path are updated so that $Q(s, a)$ represents the average !v of
all simulations that passed through it, like described in @sec_mcts.

To decide on a move to play in !s $s_p$, a search tree is initialized with the root !n
corresponding to $s_p$.
A number of MCTS iterations is performed, and the !a $a_p$ with the highest visit count is
played:
$ a_p = limits("argmax")_a (N(s_p, a)) $

In the case where the "thinking time" for each move is limited, the maximum number of
iterations depends on the speed of the !algo.
To achieve more iterations in a give time budget, the authors also implemented a
distributed version of !ago.
It utilizes multiple machines with a total of 1202 CPUs and 176 GPUs to parallelize the
search.
This distributed version was able to beat a professional human player in 5 out of 5 !gs.

]

=== !agoz
<sec_alphago_zero>

!agoz by #citet("alphago_zero") is similar to !ago:
It also uses MCTS with !nns to play Go at super-human levels.
However, the authors improved over !ago by simplifing the !arch in several aspects, while
also improving the playing strength.

The main changes over !ago are:

*Training*\
Most importantly, !agoz is trained solely by !rl and selfplay.
!agoz uses MCTS for all the selfplay during training.
This contrasts with !ago, which uses human data and supervised learning, as well as
selfplay without any search.

*!Mcts*\
The search in !agoz works the same way as in !ago, but does not use any rollouts during
the MCTS.
!agoz solely relies on the !net !v !preds to assign !vs in the search tree.

*!Net !arch*\
!agoz only uses a single CNN $f_theta$ with two output heads, predicting !p $p$ and !v $v$
together: $(p, v) = f_theta (s)$.
The input to this unified !net are images only consisting of the Go stone positions, no
extra feature planes are added.
Contrastingly, !ago uses two separate !nns for !p and !v, and the !net input contains
hand-crafted features.

The !agoz training pipeline performs these tasks in parallel:
- optimizing the !nn's parameters $theta_i$ from recent selfplay data
- evaluating selfplay !pls $alpha_theta_i$
- using the the best performing !pl so far, $alpha_theta_*$ to generate new selfplay data

Specifically, selfplay with the !pl $alpha_theta_*$ is carried out in the following
manner:
At each !s $s$ of the !g, a !mcts with 1600 iterations is performed.
The MCTS executes like in !ago with $lambda = 0$, that is, without any rollouts.
A !nn $f_theta_*$ guides the search process by !p !preds $p(a, s)$ and provides !v
estimates $v(s)$, for details see @sec_alphago.
An distinction to !ago is that the image of the Go board is randomly rotated or flipped
before using it as the input for the !nn.
This data augmentation step exploits symmetries of the !g of Go and aims to reduce !pred
bias.

The search outputs !probs $pi$ of playing each possible move, proportional to the visit
counts of the root !n in the search tree, $pi(a|s) prop N(s, a)$.
To ensure diverse !g openings, the first 30 moves are sampled from $pi$, so
$a_t tilde.op pi_t$ for $t = 0, ..., 29$.
The rest of the moves are selected according to the !a $a_t$ with the highest visit count
$a_t = limits("argmax")_a ( N(s_t, a) )$.
!Gs are played until an terminal condition is reached, and the outcome of the !g is scored
as $z$.

Dirichlet noise is blended into the !p !preds $p$ of the root !n to encourage exploration,
specifically the prior !probs $P(s, a)$ are calculated as
$ P(s, a) = (1 - epsilon) p_a + epsilon eta_a $
with $eta_a tilde.op "Dir"(0.03)$ and $epsilon = 0.25$.
Adding exploration this way ensures all moves may be tried, but the search can still
overrule bad !as.

The !nn $f_theta_i$ is trained on samples $(s, pi, z)$ drawn from the latest selfplay !gs.
Random rotations and reflections of the input image to the !nn are used to provide data
augmentation.
Mean-squared error and cross-entropy is used to align the !net's !preds 
$(p, v) = f_theta_i (s)$ with the search !probs $pi$ and !g outcome $z$, respectively.
Specifically, the !net is optimized using gradient descent on the loss !fn
$ l = (z - v)^2 - pi "log" p + c norm(theta)^2 $
where the last term is used for L2 weight normalisation.

New versions of the !nn $f_theta_i$ are evaluated in a tournament against the current
selfplay !pl $alpha_theta_*$ before they may be used for the selfplay data generation.
If the new !pl $alpha_theta_i$ wins 55% of the !gs against $alpha_theta_*$, it becomes the new
$alpha_theta_*$.
This ensures the highest quality data is generated for !nn training.

!agoz's training process starts out with random selfplay.
Over the course of 40 days and 29 million !gs, it learns Go capabilities that exceed those
of humans and its precursor !ago.


=== !az

!az, proposed by #citet("alphago_zero") is the result of minor architectural changes to
make !agoz compatible with two more board !gs, chess and shogi.
It is a step towards a general !rl !algo that masters more than one !g.

!az uses the same !algo and !nn !arch for all three !gs.
The two main changes over !agoz are:

*No data augmentation*\
!agoz's data augmentation step of rotating and flipping board !reprs is removed, because
it exploits Go-specifid symmetries.
The data augmentation provided an eightfold increase in training data, !az therefore
trains 8 times slower on the !g of Go than its precursor.

*Shared !net for selfplay and training*\
!agoz maintains separate !nns for selfplay and training, and switches them over when the
new !net performs better than the current one.
In contrast, !az only uses a single !net.
It always uses the latest !net parameters for selfplay data generation.

The publication shows that the !az training approach is feasable of generalization by
evaluating it on Go, chess and shogi.
!az outperformed existing state-of-the-art programs in all of the !g, including its
precursor !ago in the case of Go .
The training success was also reported to be repeatable across multiple independant runs.

=== !mz
<sec_muzero>

!mz by #citet("muzero") is yet another improvement over !az in terms of generalization.
!az uses a simulator during the tree search to obtain the !g !s for a hypothetical !seq of
!as.
In contrast, !mz learns a dynamic model of the !g and uses it in the !mcts to plan ahead.
In fact, this allows !mz's application to broader set of !envs including single !pl !gs
and non-zero !rs at intermediate steps.
The authors applied !mz to the !gs Go, shogi, chess and the complete Atari suite.
In the case of board !gs, it matched the performance of !az and in the Atari !env it
outperformed existing approaches.

#[

#import "drawings/muzero.typ": rep, dyn, pred, mcts, training

Just like !az, !mz uses a !net #pred for !preds inside the search tree.
However, !mz introduces two additional !nns #rep and #dyn.
The !dnet #dyn learns !s transitions of the !env:
Given a !s $s^t$ and a hypothetical !a $a^t$, #dyn<join-right> predicts the next !s
$s^(t+1)$ and !r $r^(t+1)$ the !env will respond with.
The !ss $s^t$ are encoded in a learnt latent space, so a !rnet #rep translates !obss into
!latreps.

To differentiate !ss and !as occuring in an !env !traj from those used by !net inferences,
I introduce the following notation:
!Ss and !rs occuring in a !g are marked with subscript, so the !s and !a at time $t$ are
$s_t$ and $a_t$, respectively.
A superscript is added to denote !ss predicted by the !nns, so for example
$#rep (s_t) = s_t^0$.
The superscript is increased for each inference step with the !dnet #dyn, for example
$s_t^1 = #dyn (s_t^0, a^0)$.
In some cases, the subscript may be omitted to simplify notation.

The !dnet can be applied recurrently, so any !r $r^n$ and !s $s^n$ $n$<join-right> time
steps ahead can be predicted given an initial !s $s^0$ and a series of hypothetical !as
$a^0, a^1, ..., a^n$:
$ (s^t, r^t) = #dyn (s^(t-1), a^t) $
This process is used during !mcts, as visualized in @fig_muzero_mcts.
To decide on an !a in !g !s $s$, a search tree is initialized with the !g !obs:
$s^0 = #rep (s)$.
During the MCTS expansion phase, the !dnet #dyn is used to obtain the next !s $s^(n+1)$
and the !r $r^(n+1)$ associated with the !s transition for an !a $a^n$ in !s $s^n$.

The MCTS simulation phase executes like in !az, where #pred predicts a !v estimate and
search !p of a newly expanded !n $s^n$:
$(v^n, p^n) = #pred (s^n)$

#figure(
  mcts,
  caption: [!Mcts in !mz]
) <fig_muzero_mcts>


Like !az, !mz uses selfplay with MCTS to generate training data.
For each selfplay step at time $t$ the data $(s_t, a_t, r_t, pi_t, z_t)$ is recorded.
The first three entries contain the !s of the !env, executed !a and received !r,
respectively.
$pi_t$<join-right> denotes the search !p, like in !mz's precursors.
For details see @sec_alphago_zero.
$z_t$<join-right> is the RL sample !ret as introduced in @sec_rl.
In the case of board !gs, it is equal to the final outcome of the !g $z$, like in !mz's
precedessors.
For Atari !gs, $z_t$<join-right> is calculated to be a n-step !ret.

#[

#let series(x, end: $K$) = $#x _t, #x _(t+1), ..., #x _end$
#let loss(letter, target, pred, start: 0) = $sum_(k=start)^K l^letter (target _(t+k), pred _t^k)$

During training, the !dnet is unrolled for $K$ steps and aligned with a !seq of !env !ss,
!as and !rs from the selfplay data.
Specifically, a training example beginning at time step $t$ consists of the tuple
$(s_t, (series(a)), (series(r)), (series(pi)), z_t)$, where $K$ is the unroll length, a
hyperparameter.
The process is illustrated in @fig_muzero_training for $K=2$ and $z_t = z$ for all $t$.

#figure(
  training,
  caption: [
    Training setup in !mz in the case of board !gs, unrolled for~2 steps.
    Gray arrows indicate training losses.
  ]
) <fig_muzero_training>

First, a !latrep $s_t^0$ is obtained using the !rnet: $s_t^0 = #rep (s_t)$.
Then, the !dnet #dyn is applied recurrently $K-1$ times with the !as from the !traj
$series(a)$.
This yields !latreps for the !ss $s_t^1, s_t^2, ..., s_t^K$ and corresponding !r !preds
$r_t^1, r_t^2, ..., r_t^K$.
At each unroll step $n = 0, 1, ..., K$, the !pnet #pred predicts a !v and !p $(v^n, p^n)$
from the !latrep $s_t^n$.

Losses for the !r, !p and !v !preds are calculated at each unroll step to align the !net
!preds with the actually occurred !rs $r_t$, search !ps $pi_t$ and sample !rets $z_t$.
Specifically, the total loss $l_t$ for an unroll !seq starting at $s_t$ and with length
$K$ is given by
$ l_t = loss(p, pi, p) + loss(v, z, v) + loss(r, r, r, start: 1) $
with $l^p$, $l^v$<join-right> and $l^r$ being loss !fns for !p, !v and !r, respectively.
The parameters of the three !nns are updated jointly via backpropagation on the total loss
$l$.

]

]
