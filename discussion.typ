= Discussion
/*
Hier diskutierst du deine Ergenisse und deine Arbeit.
Hier darfst du jetzt Sachen in deine Ergebnisse interpretieren (z.B. Algorithmus A ist
besser als Alg. B).
Außerdem darst du hier eigene Gedanken reinbrigen. Z.B. warum du denkst das bestimmte
Sachen funktioniert haben bzw. nicht funktioniert haben.
Ich finde es schön, mögliches Future Work am Ende der Discussion zu diskutieren (manche
andere schreiben das auch am Ende der Conclusion.)
*/
#import "thesis.typ": citet

In this chapter I discuss the results presented in @sec-results.

== Ablation Study: Latent Loss Variants

Here I provide a discussion about the results of the ablation study for the latent loss
variants, as presented in @sec-results_ll.

At first glance, @fig-plot_ll shows similar training dynamics among the three variants:
All variants eventually learn to play the !g with near-perfect accuracy, as indicated by
the almost constant score of 1 by the end of the training.
Accordingly, based on the final scores in @tbl-scores_ll, no best or worst performing
variant can be identified with statistical significance.

In @fig-plot_ll, the !effz variant learns the fastest, given the fast rise of the score
$s$ between $-0.5 < 0.5$ curve, beginning at $n=12000$ !env steps.
Compared to the !mz variant with no latent loss, this is the expected result and
qualitatively aligns with the findings reported by #citet("effzero").
However, learning slows again when the score approaches 1.
Overall, no practical speedup in terms of reaching the final accuracity is achieved.

#[
#import "drawings/muzero.typ": rep, dyn
My proposed symmetric latent loss does not perform any different than !mz.
On one hand, I did expect my variant to perform at least more similar to !effz, because
the additional latent loss provides feedback to both the !repr #rep and !dnet #dyn.
However, the findings suggest that the symmetric latent loss might provide detrimental
training signals to the !rnet #rep.

On the other hand, no latent collapse is observed without the stop-gradient operation, as
would be evident by very bad playing performance.
This is in line with my theory that the other training losses achieve sufficient
decorrelation of the latent space to prevent a collapse, as outlined in
@sec-mod_symm_latent_loss.
]
== Ablation Study: !TNs

Here I discuss the results of my ablation study as presented in @sec-results_tns.

@fig-plot_tns shows that the two variants perform differently.
Specifically, the variant without !tns learns faster, it reaches the optimal !g score of 1
after 10k !env steps.
In contrast, the variant with !tns learns slower and does not reach perfect play by the
end of 25k !env steps, as shown in @tbl-scores_tns.
However, a direct interpretation of the final selfplay scores is difficult because the
variance of the runs is very high in comparison with the difference of in scores.

My original hypothesis was that !preds for !ns beyond !g end might lead to bias in the !n
!vs of the tree.
I assumed that this bias slows down learning, so the observed result is surprising to me.

The results of manually inspecting the search trees reveal several interesting aspects:
First, in the variant without !tns, there is in fact bias in the !n !v estimates, as
evident by deviations from the optimal !n !vs and !n !vs outside of the range $[-1, 1]$ of
possible !vs for the !g Catch.
Second, nonzero !r !preds for !ns beyond !tss are very likely the cause of the observed bias:
I come to this conclusion, as there is no other way of creating !n !vs $v$ with magnitudes
$|v| > 2$ in the tree, given correct !r !preds for !ns correspondig to !g !ss.
Third, the nonzero !r !preds appear at depth $K - 1$ in the tree, indicating that the
training of absorbing !ss does not generalize well for unroll counts higher than the one
used during training, $K$.
Finally, the variant with !tns correctly predicts !tns and the search does not expand beyond
them.

From the faster learning with !tns, I conclude that the bias in the search may actually help
convergence in the specific !env tested.
The larger magnitudes of the !n !vs might pronounce !v differences for distinct !as,
helping the pUCT selection formula focusing on more promising !ns.

An interesting direction for future research might be to further explore the effects of
predicting !tns in different types of !envs.
