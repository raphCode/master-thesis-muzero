#let abbrev(abb, text) = body => {
  show regex("\b" + abb + "\b"): text
  body
}

#let general(body) = {
  show "muzero": [MuZero]
  show: abbrev("rl", "reinforcement learning")
  body
}

#let player(body) = {
  show: abbrev("sp", "single player")
  show: abbrev("2p", "two player")
  show: abbrev("mp", "multiplayer")
  body
}
