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

In this chapter I discuss the results from the previous chapter.

== Ablation Study: Latent Loss Variants

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
