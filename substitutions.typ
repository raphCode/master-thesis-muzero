#let abbrev(abb, text) = body => {
  show regex("\b" + abb + "\b"): text
  body
}

#let abbrev_plural(abb, text) = {
  let plural_form = {
    if text.match(regex("[^aeiou]y$")) != none {
      text.replace(regex("y$"), "ies")
    } else {
      text + "s"
    }
  }
  body => {
    show regex("\b" + abb + "s?\b"): match => {
      if match.text.trim(abb, at: start) == "s" { plural_form } else { text }
    }
    body
  }
}

#let general(body) = {
  show "muzero": [MuZero]
  show "azero": [AlphaZero]
  show "effzero": [EfficientZero]

  show: abbrev("rl", "reinforcement learning")
  show: abbrev("mcts", "monte carlo tree search")
  show: abbrev_plural("impl", "implementation")
  show: abbrev_plural("algo", "algorithm")
  show: abbrev_plural("pred", "prediction")
  show: abbrev_plural("latrep", "latent representation")

  show: abbrev_plural("net", "network")
  show: abbrev("rnet", "representation net")
  show: abbrev("pnet", "prediction net")
  show: abbrev("dnet", "dynamics net")
  body
}

#let player(body) = {
  show: abbrev("sp", "single player")
  show: abbrev("2p", "two player")
  show: abbrev("mp", "multiplayer")
  body
}

#let game(body) = {
  show: abbrev("0sum", "zero-sum")
  show: abbrev_plural("traj", "trajectory")
  body
}
