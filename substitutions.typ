#let abbrev(abb, text) = body => {
  show regex("\b" + abb + "\b"): text
  body
}

#let general(body) = {
  show "muzero": [MuZero]
  show: abbrev("rl", "reinforcement learning")
  body
}
