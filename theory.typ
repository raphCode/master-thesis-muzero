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
