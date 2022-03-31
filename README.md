# My Master's Thesis

**Deep Reinforcement Learning with MuZero:
Theoretical Foundations, Variants, and Implementation
for a Collaborative Game**

# Repo organisation

This repo contains the source code of:
- my custom MuZero implementation ([branch: master](../../tree/master/))
- the actual thesis paper and the presentation slides, written in
  [Typst](https://typst.app) ([branch: paper-typst](../../tree/paper-typst/))

The repo is mostly a show-off of the work I did for my thesis, do not expect any more
value or useable software.

## Other branches

- __paper-*__: old versions of the paper source code, in different typesetting systems.
  I started in LaTeX [(branch)](../../tree/paper-latex/), then tried
  [Sile](https://sile-typesetter.org/) [(branch)](../../tree/paper-sile/), before I
  discovered Typst.
- __wip*__: Various work-in-progress side branches of the MuZero implementation, most to
  all work should also be in master. Numbered according to age.
- _all other branches_: Various older checkpoints of the MuZero implementation, not really
  interesting except for understanding the development history.

## Commit History

I made heavy use of history-rewriting to tidy up commit diffs and messages.
That means commit dates may be all over the place (but author dates are accurate).

Near the end, I biseced some performance regressions, and had to fix bugs in very old
commits.
Fixing these bugs required rewriting the history of the master branch, and I never rebased
the wip branches, so their history does not line up anymore nicely.
