= Theory
<theory>
/*
Ich finde ein extra Theorie Kapitel gut, in dem man die theoretischen Grundlagend
zusammenfasst, die nötig sind um die verwendeten Algorithmen zu verstehen. 
In Research-papern ist hierfür meist kein Platz aber in einer Abschlussarbeit kann man
damit gut zeigen, dass man das Material verstanden hat.
*/

#import "thesis.typ": citet

- RL terms and basics
  - model-based
  - planning
  - !mcts

== !RL Background

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
$ p(s', r | s, a) eq.def Pr{s_(t+1)=s', r_(t+1)=r | s_t=s, a_t=a} $
for all $s', s in S, r in R, a in A(s)$.

It describes the !prob of the !env ending up in !s $s'$ and yielding !r $r$ when executing
!a $a$ in !s $s$.
@sutton

Importantly, !mdps can also capture the aspect of !as affecting not only the immediate !r,
but also subsequent !ss and thus future !rs.
In the example of certain !g, like tic-tac-toe or chess, !rs are not given after each
move.
Instead, they come at the end of the entire !g in the form of a win or loss.
Obviously, this final !r depends on most or all moves during the !g.
@sutton

In the general case, when a !r is influenced not only by its immediately preceding !a, but
also by older ones, the situation is called a delayed !r.
Delayed rewards pose a challenge in !rl:
Since !as have long-term consequences, the !ag must decompose which of his !as contributed
to the final !r and how.
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
For this purpose, a hyperparameter #sym.gamma is introduced, called the !df.
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
The !exret describes the expected value of the !ret $G_t$ in !s $s_t$, if the !p #sym.pi
is followed.
It is therefore also called the !v $v_pi(s)$ and defined as
$ v_pi(s) eq.def EE_pi [G_t | s_t=s] = EE_pi [sum_(k=0)^(n-1) gamma^k r_(t+k+1)
#move(dy: -3pt, scale(y: 300%, $|$)) s_t=s] $
for a !s $s_t$ when following !p #sym.pi.
@sutton

The !v is thus an estimate of how good it is for the !ag to be in a particular !s,
measured by the objective !fn, the !ret.
@sutton

In !ts, the !v is always zero per definition:
There are no future !rs possible.
@sutton

To estimate the !v !fn, we can use data collected through interactions with the !env.
For example, suppose an !ag pursues a !p #sym.pi and experiences several !epis where the
same !ss occur repeatedly.
The !ag then collects all the !rets that have ever occurred for all !ss and averages them
for each !s.
These averages would converge to the optimal !v !fn $v_pi$ as the number of times each !s
is visited approaches infinity.
Averaging over many random samples of actual data in this way is referred to as a !mc
method.
@sutton

== !GT

This section introduces basic !gtic concepts, just enough to provide justification for the
design and behavior of !mz.

=== Overview

!Gt is a broad interdisciplinary field that uses mathematical models to analyze how
individuals make decisions in interactive situations.
It is commonly used in economics, but has other widespread applications.
@gtheo

In this thesis we are interested in !gtic analysis of the behavior and interaction of
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

Note that the concepts are comparable to !rl, but have slightly different terminology.
@concept_comparison shows a comparison of the analogous terms between the two fields.

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
) <concept_comparison>

!Gt may differentiate !gs according to different properties which are explained in the
next chapters.

=== !ZSUM

!Zsum !gs are !gs in which the sum of all !pls' payoffs equals zero for every outcome of
the !g.
It is a special case of the more general concept of constant-sum !gs, where all payoffs
sum to constant value.
In other words, a !pl may benefit only at the expense of other !pls.
Examples of !2p !zsum !gs are tic-tac-toe, chess and Go, since only one !pl (payoff of 1)
may win, while the other looses (payoff of -1).
Also poker is considered a !zsum !g, because a !pl wins exactly the amount which his
opponents lose.

Contrastingly, in non-!zsum !gs, no restriction is imposed on the payoff vectors.
This situation arises when modeling real-world economic situations, as there may be
gains from trades.

=== !I

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

=== Simultaneous / !Seql Moves

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
These !gs are also called dynamic !gs or extensive form !gs.
To distinguish them from simultaneous !gs, a !pl making a decision must have !i about the
previous decisions of other !pls.
@gtheo2

This can be illustrated by a variant of !rps:
!Pls now have to write down their choices without revealing them to each other.
Then their (binding) choices are uncovered and the outcome of the !g can be determined.
While this !g can be played sequentially, with !pls choosing their !as at different times,
the simultaneous nature of the !g remains.
At best, it can be seen as a sequential but incomplete information variant of !rps.
@gtheo2

=== Cooperation

!Gt can be divided into two branches, cooperative and non-cooperative.
The cooperative approach studies !gs where the rules are only broadly defined.
In fact, the rules are kept implicit in the formal specification of the !g.

Since the rules are not specific enough to analyze individual decision making, cooperative
!gt looks instead at coalitions of !pls.
These coalitions assume that !pls can commit to work together through binding agreements
or through the transfer of decision-making power.

Overall, cooperative !gt provides a framework for understanding how different parties can
work together toward common goals.

Unlike the other branch, non-cooperative !gt requires an exact specification of the rules
of the !g.
For this reason, it is also known as the theory of !gs with complete rules.
It allows an analysis of the individual decisions of !pls without relying on commitment
devices.
!Pls are assumed to act solely out of self-interest, i.e. to choose !as that maximize
their payoff.

However, this can lead to seemingly cooperative behaviour in some settings:
!gs may have rules that require !pls to work together to achieve a mutually beneficial
goal.
This goal cannot be achieved by one !pl alone, so !pls are motivated to behave
cooperatively.

The !gs studied in this thesis are strictly non-cooperative in a !gtic sense.
To avoid any confusion, it is important to point out that this thesis refers to !gs being
cooperative when:
- all !pls act out of self-interest and
- the rules of the !gs are designed to encourage cooperation.
The latter is achieved by designing a !g outcome with a large payoff for all !pls which
can only be reached by working together in a way intended by the rules.

- previous versions
  - AlphaGo
  - AlphaGo Zero
  - AlphaZero
  - MuZero

- original MuZero limitations
  - !pinf !gs
  - !seql !gs
  - single !pl / two !pl !zsum
  - unique best strategy must exist (counterexample: starcraft)
