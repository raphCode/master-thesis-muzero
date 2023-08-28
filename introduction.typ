= Introduction
<introduction>
/*
Hier beschreibst du die Idee und Motivation deiner Arbeit. 
Ziel ist es zu sagen was du gemacht hast und warum es wichtig ist, dass es jemand
gemacht hat.
Es sollte reichen die Introduction und Conclusion zu lesen, um vollständig zu verstehen
was du gemacht hast und was rausgekommen ist (ohne details natürlich).
Laut Prof. André eines der wichtigesten Kapitel.
*/

#import "thesis.typ": citet

In 2016, !ago was unveiled - the first computer program capable of playing the complex
strategy !g of Go at a level surpassing human masters.
To understand why this is a remarkable achievement, let's summarize how most traditional
!algos approach gameplay.


Traditional !algos attempt to identify an optimal !a by simulating numerous future moves
and selecting the one with the best potential outcome.
The underlying assumption is that examining enough moves will eventually reveal a strong
move by chance.
However, Go poses a significant challenge because of its expansive 19x19 grid,
considerably larger than, for example, a chess board.
Consequently, the branching factor is very high in the !g tree: the number of possible
outcomes escalates with each additional move searched ahead.
To have a reasonable chance of discovering an appropriate move by chance, a prohibitively
large number of options must be evaluated.
As a result, traditional search approaches are computationally infeasible.

Even with an appropriate search approach, there remains the problem of evaluating specific
board conditions.
An accurate assessment of positions is necessary to guide the search towards victory.
Go !gs can go on for hundreds of moves, making it hard to tell which positions are good or
bad for either !pl in the long run, especially since a single move can have a significant
impact on the rest of the !g.

Nevertheless, in 2018, #citet("azero") advanced upon this technology with !az, which was
able to master three different board !gs: Go, Chess and Shogi.
This is notable given that previous !algos relied heavily on specific domain-related !i,
such as the individual !g rules.
The ability of a single !algo to excel in several distinct !envs highlights its
versatility~- a key goal of !rl.

Like most search approaches, !az uses a simulator that must be explicitly programmed with
the !env rules to determine the subsequent !s following an !a.
This is not a limitation per se, in fact it is sometimes helpful or easier in some
real-world applications.
For example, #citet("azero_sorting") made use of the !az !arch to discover a faster
sorting !algo.
In this particular case, the !env is presented as a !g to modify an assembly program in
incremental steps.
The correctness and speed of the resulting program is measured to calculate the !r.
An explicit simulator is a good fit here, since it guarantees that searched !as and their
!rs are accurate, even for paths that have never been visited before. @azero_sorting

In other scenarios, it is more difficult to provide a good !impl for the !env during the
search.
For instance, #citet("muzero_vp9") optimized video encoding to save bandwidth while
preserving the visual quality.
In this case, the task is framed as a !g for a !rl !ag, which has the ability to adjust
codec parameters.
Therefore, video encoding and evaluation of the resulting visual quality must be managed
in the !g tree search. @muzero_vp9

!mz @muzero is a further evolution of !az.
The major difference is that it removes the need for an explicit simulator in the search
phase.
Instead, !mz develops its own internal model to predict future !env !ss.
In effect, it learns how to play !gs without having access to the underlying rules or
dynamics.
This way, even the above example of optimizing video encoding can be mastered.
Perhaps surprisingly, !mz also matches !az's performance in the aforementioned board !gs.
Moreover, it also performs well on the Atari suite of 57 video !gs, demonstrating the
ability to generalise across many domains. @muzero

While !mz's capabilities are impressive, the original !impl is limited in a number of
ways.
First, it was designed for one- or !2p !gs only.
For !2p !gs, an additional requirement is that the !g must be !zsum, meaning that one
!pl's loss is the other's gain.

The original !impl also can only handle deterministic !envs.
This excludes its application to games with chance events, such as dice throws.

Finally, !mz has high computational requirements, which results in long training times.

This thesis is structured as following:\
First, theoretical foundations of !rl, !gt and computers playing !gs are outlined.
They lead up to an understanding of the !mz !algo and its precursors.
After that, related !algos are reviewed, highlighting their differences, strengths and
limitations.

Builing on this study of prior work and theoretical foundations, I propose:
- an extension of !mz to !mp !gs including chance events and arbitrary !rs.
- performance enhancements to improve the training time.

This thesis is accompanied by an !impl of !mz that includes modifications as proposed by
me and in prior work.
With this !impl, the modifications are experimentally evaluated on two aspects:
First, an ablation study compares the influence of the different performance enhancements.
Second, I confirm my !mp extension is able to learn a cooperative !mp !g.
