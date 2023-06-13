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

In the last years, muzero has been cited as a remarkable breakthrough in the field of rl,
being the first algo that beats human experts in the game of Go.
A computer mastering a particular game may not look like much, after all computers got
very good at playing many other complex games, for example chess.
In order to understand why Go is different, one first needs to understand how computers
usually play games:
Traditional algos try to find a good move by considering different moves and the
opponent's likely responses.
After simulating many moves, the one with the best future outcome for the computer's
player is selected.
To obtain a useful result from such a search approach, a substantial part of the future
game tree must be explored.
This is for two reasons:
First, because moves are often picked at near-random, a high number of moves should be
visited to make sure a good move is uncovered.
Second, the quality of a move is often revealed many moves later, in extreme cases not
until the end of the game.
This requires the search to reach very deep in the future to avoid misjudging a certain
move.

Go provides a big challenge to such an approach due to its large 19x19 grid board and the
fact that games can go on for hundreds of moves.
The large board allows many possible moves at each point in the search, resulting in a
very large search space.
Worse, the large length of the games requires very deep searches before meaningful results
can be obtained.

In fact, the Go search space is considered intractable because exploring it in a search
exceeds the currently available computing budget by orders of magnitudes.
This suggests that muzero's success in Go must be due to a effective algorithmic approach
rather than raw computing power.

Furthermore, muzero is designed to be applicable to more than just Go.
The original paper also successfully applies muzero to chess, shogi and the Atari suite of
games, each of which are considered challenging games.
This is possible since the actual algo does not include any domain-specific knowledge,
that is, no Go- or chess-specific information is built in.
This includes even the game's rule, which the algo learns on its own.
Compared to traditional approaches, it might come as a surprise that providing the algo
with less information actually leads to better performance.

The performance and generality of muzero is already impressive.
However, the original impl is limited in a number of ways.
First, it was designed for one- or two-player games only.
In the case of 2p games, an additional requirement is that the game is 0sum, meaning
that one player's loss is the other's gain.

muzero also only works in perfect information games.
This means that the game does not conceal any information from the player making a
decision, he is perfectly informed about all previous events.
For example, in chess the current state of the board provides this information.
On the other hand, in card games like poker, information about other players' hands is
hidden.

While the above discussion refers to games, the muzero algo has also been successfully
applied to practical problems.
One example is adjusting codec parameters during video compression, where it outperformed
previous approaches. @vp9_muzero
Likewise, the aforementioned restrictions of compatible games also translate into
limitations of real-world applications.
The original muzero algo is thus unsuitable for environments with multiple agents,
arbitrary scoring functions, or imperfect information.
