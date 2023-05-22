= Approach
<approach>
/*
Hier beschreibst du, was du in der Arbeit neues gemacht hast, und wie du es implementiert
hast.
*/

- MuZero implementation
  - mcts behavior customizeable
    - node selection
    - policy calculation
    - action selection
  - supports arbitrary games
  - pytorch
  - typed codebase, passes mypy
- extensions / variants:
  - efficient zero
  - terminal states
  - chance states
  - multiplayer support: teams, prediction of current player
- interplay of mcts policy and selection function
  - unstable behavior of original setup: UCT scores and visit count policy
- application to carchess

