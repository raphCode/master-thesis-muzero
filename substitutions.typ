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
  show: abbrev("rl", "reinforcement learning")
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
  body
}

