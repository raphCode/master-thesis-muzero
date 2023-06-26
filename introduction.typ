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

In 2016, alphago was unveiled - the first artificial intelligence program capable of
playing the complex strategy game of Go at a level surpassing human masters.
To understand why this is a remarkable achievement, let's summarize how most previous
algorithms approach gameplay.

Traditional methods try to find a good action by simulating a large number of future moves
and selecting the one with the best possible outcome.
The assumption is that by examining enough moves one will eventually find a strong move.
However, Go presents a major challenge due to its extensive 19x19 grid, which is
considerably larger than, say, a chessboard.
This leads to a very high branching factor in the game tree search, rendering a classical
search approach computationally impossible, as the number of possible outcomes escalates
with each additional move evaluated.

Even with an appropriate search approach, there remains the problem of evaluating specific
board conditions.
An accurate assessment of positions is necessary to guide the search towards victory.
Go games can span hundreds of moves, making it difficult to determine which positions are
good or bad for either player in the long run, especially since a single move can have a
significant impact on the rest of the game.

alphago's success suggests that it uses an intelligent algo, as the improvements cannot be
attributed solely to increased computational power.

Nevertheless, in 2018, #citet("azero") advanced upon this technology with azero, which was
able to master three different board games: Go, Chess and Shogi.
This is notable given that previous algos relied heavily on specific domain-related
information, such as the individual game rules.
The ability of a single algo to excel in several distinct envs highlights its
versatility~- a key goal of rl.

Like most search approaches, azero uses a simulator that must be explicitly programmed
with the env rules to determine the subsequent state following an action.
This is not a limitation per se, in fact it is sometimes helpful or easier in some
real-world applications.
For example, #citet("azero_sorting") made use of the azero arch to discover a faster
sorting algo.
In this particular case, the env is presented as a game to modify an assembly program in
incremental steps.
The correctness and speed of the resulting program is used to calculate the reward.
An explicit simulator is a good fit here, since it guarantees that searched paths and
their rewards are accurate, even for paths that have never been visited before.

In other scenarios, it may be more difficult to provide a good impl for the env during the
search.
muzero addresses this challenge by eliminating the need for a simulator in the search
phase.
Instead, it develops its own internal model to predict future env states.
By doing so, it can learn how to play games without having access to the underlying rules.
Perhaps surprisingly, this allows muzero to match Azero's performance in the
aforementioned board games.
Moreover, it also masters the Atari suite of 57 video games, demonstrating the ability to
generalise across many domains.

It should come as no surprise that muzero has also been applied successfully in the real
world.
For example, #citet("muzero_vp9") used it to achieve bandwidth savings during video
streaming.
In this scenario, muzero adjusted parameters of a video codec to reduce the file size
while maintaining visual quality.
Using muzero's learned model of the env eliminates the need to modify the video encoder to
work with the tree search.

While muzero's capabilities are impressive, the original impl is limited in a number of
ways.
First, it was designed for one- or 2p games only.
For 2p games, an additional requirement is that the game must be zsum, meaning that one
player's loss is the other's gain.

muzero also only works in perfect information games.
The term means that the game does not hide any information from the player making a
decision, he is perfectly informed about all previous events.
In chess, for example, the current state of the board provides this information.
On the other hand, in card games such as poker, information about other players' hands is
hidden.

Finally, the algo has high computational requirements, which results in long training
times.

This thesis reviews the design of muzero and provides an impl of the muzero algo.
The impl includes modifications to improve computational requirements and performance.
The modifications come from previous work on muzero and azero, as well as my own, and are
evaluated using an ablation study.

I also propose modifications to extend the muzero algo to multiplayer games with arbitrary
reward functions.


