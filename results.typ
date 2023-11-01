= Results
/*
Hier beschreibst du die Ergebnisse deiner Evaluation descriptiv. 
Hier gibst du noch keine eigene Meinung oder Interpretation der Ergebnisse. Wirklich nur
die Fakten.
*/

#import "thesis.typ": neq

#let steps = 1000
#let ablation_plot_caption(n) = [
  Comparison of the selfplay scores across variants, averaged over #n runs each.
  Error bands show 90% confidence intervals.
]

In this section I report the results of my training runs.

#[
#let n = 50000
#let g = 10
The x-axis of all plots shows the number of steps $n$ taken in the !env.
For plotting, the number of data points has been resampled to #steps steps.

For example, if a !g takes #g steps, a datapoint for the score is recorded every #g steps.
If the training is performed for #n total steps, this generates a total of #{n/g} data
points.
The resampling picks #steps equidistant samples from these data points, so that data
points are plotted in increments of #{n/steps} !env steps.
The specific resampling !algo I use is the reservoir sampling !impl from tensorboard.
]

== Ablation Study: Latent Loss Variants

#[
#import "experiment_data/nruns.typ": nruns_latent_loss as nruns

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
]

#pagebreak(weak: true)
== Ablation Study: !TNs

#[
#import "experiment_data/nruns.typ": nruns_latent_loss as nruns

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
]
