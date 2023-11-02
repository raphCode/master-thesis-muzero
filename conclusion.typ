= Conclusion
/*
Hier fasst du nocheinmal kurz die wichtigsten Erkenntnisse und Ergebnisse deiner Arbeit
zusammen.
MAXIMAL 1 Seite.
*/

#import "thesis.typ": citet

In this thesis, I analyze !mz, a !rl !algo that learns a model of the !env and uses it to
plan ahead in a !mcts (MCTS).

I outline the limitations of the original !impl, in particular how it is restricted to
deterministic single-!ag !envs and !zsum !2p !gs.
The main contribution is an extension of !mz to more general !envs with any number of !ags,
stochasticity, and general !rs.

For this, I modify the !env model such that it learns which !ag can make a decision at any
given !s of the !env.
Also, I replace the negamax MCTS of the original !mz !impl with maxn MCTS, so that the
learned behavior approaches the !gtic optimal solution for all !ags involved.

I implement and successfully evaluate my proposed !arch on the collaborative !mp !g
Carchess.
My !impl is capable of learning this !g, and achieves a reasonable score.

I also propose two additional modifications of !mz.
First, a modification of the MCTS to handle !tss during search.
Second, the introduction of a symmetric latent loss, building on the work of !effz by
#citet("effzero").

For each modiﬁcation, I perform an ablation study to investigate its effect:
In the specific !env and settings I use, the two modifications perform worse than (in the
case of !tns) and indistinguishable from (in the case of the symmetric latent loss) the
original !mz baseline.
