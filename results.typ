= Results
<sec-results>
/*
Hier beschreibst du die Ergebnisse deiner Evaluation descriptiv. 
Hier gibst du noch keine eigene Meinung oder Interpretation der Ergebnisse. Wirklich nur
die Fakten.
*/

#import "thesis.typ": neq

#let resample = 1000

#let ablation_plot_caption(n) = [
  Comparison of the selfplay scores across variants, averaged over #n runs each.
  Error bands show 90% confidence intervals.
]
#let ablation_scores_caption(n, steps, samples) = [
  Mean and standard deviation of the score during selfplay after #steps !env steps,
  averaged over #n runs and last #samples data points each.
]
#let ablation_score_text(n, steps, samples, a, b) = [
  shows the final scores after #steps !env steps.
  The mean and standard deviation are calculated over the last #{n * samples} !g outcomes
  #footnote[in the resampled time domain]
  distributed over all runs. 
  Specifically, each run contributes #samples data points $z_t$, where
  $#{resample - samples} <= t < resample$.
  This corresponds to the (resampled) data points at !env steps $#a <= n < #b$.
]

In this section I report the results of my training runs.

#[
#let n = 50000
#let g = 10
The x-axis of all plots shows the number of steps $n$ taken in the !env.
For plotting, the number of data points is resampled to #resample steps.

For example, if a !g takes #g steps, a datapoint for the score is recorded every #g steps.
If the training is performed for #n total steps, this generates a total of #{n/g} data
points.
The resampling picks #resample equidistant samples from these data points, so that data
points are plotted in increments of #{n/resample} !env steps.
The specific resampling !algo I use is the reservoir sampling !impl in tensorboard.

The use of $t$ in this chapter always refers to the resampled time domain, that is
$t in NN and t < resample$.
]

== Ablation Study: Latent Loss Variants
<sec-results_ll>

#[
#import "experiment_data/data.typ": nruns_latent_loss as nruns, nsteps_latent_loss as nsteps

Here I report the results of the ablation study for the different latent loss variants.
Specifically, @fig-plot_ll shows a direct comparison of the score during selfplay for each
variant.

The plot contains three curves, one for each latent loss variant.
Each curve is the result of averaging over #nruns runs with different random
seeds.

Since Catch has a a binary outcome $z in [-1, 1]$, each plotted data point $x'_t$ at time
step $t$ is smoothed with an exponential moving average:
#neq[
$ x'_t = limits(sum)_(i=0)^t 0.97^i z_(t-i) $
<eq-score_smoothing>]
where $z_t$ is the !g outcome at time $t$.

#figure(
  image("experiment_data/latent_loss/plot.svg"),
  caption: ablation_plot_caption(nruns),
) <fig-plot_ll>

@tbl-scores_ll #ablation_score_text(nruns, nsteps, 100, 45000, 50000)

#figure(
  table(
    columns: (auto, auto),
    [*Variant*], [*Final score $mu plus.minus sigma$*],
    [!mz], [1.00 $plus.minus$ 0.0],
    [!effz], [0.96 $plus.minus$ 0.28],
    [Symmetric Latent Loss], [0.96 $plus.minus$ 0.26],
  ),
  caption: ablation_scores_caption(nruns, nsteps, 100),
) <tbl-scores_ll>

]

#pagebreak(weak: true)
== Ablation Study: !TNs
<sec-results_tns>

#[
#import "experiment_data/data.typ": nruns_terminal_nodes as nruns, nsteps_terminal_nodes as nsteps

#let nruns = 4

Here I report the results of the ablation study for !tns in the search tree.
Specifically, @fig-plot_tns shows a direct comparison of the score during selfplay for
each variant.

The plot contains two curves, one for runs with !tns, and one for runs without !tns.
Each curve is the result of averaging over #nruns runs with different random
seeds.
Smoothing is applied as outlined in @eq-score_smoothing.

#figure(
  image("experiment_data/terminal_nodes/plot.svg"),
  caption: ablation_plot_caption(nruns),
) <fig-plot_tns>

@tbl-scores_ll #ablation_score_text(nruns, nsteps, 10, 24750, 25000)

#figure(
  table(
    columns: (auto, auto),
    [*Variant*], [*Final score $mu plus.minus sigma$*],
    [Without !TNs], [0.85 $plus.minus$ 0.53],
    [With !TNs], [0.90 $plus.minus$ 0.44],
  ),
  caption: ablation_scores_caption(nruns, nsteps, 10),
) <tbl-scores_tns>

Additionally, I present an analysis of the !mc search tree of both variants near the end
of training.
Specifically, I pick multiple logged selfplay !gs at random from the last 100 selfplay !gs
in each run.
I then manually inspect the final search tree, that is, after all MCTS iterations have
passed.

The key aspects observed across multiple search trees are summarized below.

*For both variants:*
- !r !preds for all !ns correspondig to !g !ss are zero, except for the !tss, where the
  !preds are near 1 or -1.

*For the variant with !tns:*
- !tns are present in the search tree, corresponding to !tss of the !g
- all !n !vs are found to be in the interval $[-1, 1]$

*For the variant without !tns:*
- !n !vs $v$ with magnitudes $|v| > 2$ are present in the tree
- no !tns are present and the search progresses beyond !ns corresponding to !tss of !g
- !ns $>=3$ levels beyond !g end have nonzero !r !preds
]

#pagebreak(weak: true)
== Application to Carchess

#[
#let nsteps = 824584
#let w = 30

Here I report the results of the applications of my !impl of !mz to Carchess.
@fig-plot_carchess shows the score during a single run of selfplay over #nsteps !g steps
on the map "tutorial".
I define the score in Carchess as the cumultative !r as experienced by a single !pl.

At each time step $t$ in the plot, the mean and 95% confidence interval is computed over
the last #w data points.
The plot thus shows the average score, as computed over a sliding window of size #w.

#figure(
  image("experiment_data/carchess/plot.svg"),
  caption: [
    Mean score during a single run of selfplay. The error band shows a 95% confidence
    interval of the last #w scores at each data point.
    ],
) <fig-plot_carchess>

The mean and standard deviation of the selfplay score over the last 500 data points is
*$61.02 plus.minus 15.54$*.
This corresponds to !gs played during !env steps $#{nsteps / 2} <= n < #nsteps$.

]
