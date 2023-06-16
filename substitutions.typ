#let capitalize(text, first_only: false) = {
  let args = { if first_only { (count: 1) } else { (:) } }
  text.replace(regex("\b[[:alpha:]]"), match => { upper(match.text) }, ..args)
}

#let capitalize_substitution(abb, text) = {
  if abb == upper(abb) and abb.codepoints().len() > 1 {
    capitalize(text, first_only: false)
  } else if abb.match(regex("^[[:upper:]]")) != none {
    capitalize(text, first_only: true)
  } else {
    text
  }
}

#let showrule_regex_captures(pattern, callback) = body => {
  // apply a show rule to the regex,
  // match it again to get capture groups and call the callback with them
  let r = regex(pattern)
  show r: it => { callback(..it.text.match(r).captures) }
  body
}

#let abbrev(abb, text) = {
  showrule_regex_captures(
    "\b((?i)" + abb + ")\b",
    matched_abb => {
      capitalize_substitution(matched_abb, text)
    }
  )
}

#let abbrev_plural(abb, text) = {
  let plural_form = {
    if text.match(regex("[^aeiou]y$")) != none {
      text.replace(regex("y$"), "ies")
    } else {
      text + "s"
    }
  }
  showrule_regex_captures(
    "\b((?i)" + abb + ")(s?)\b",
    (matched_abb, plural_s) => {
      let sub = if plural_s != "" { plural_form } else { text }
      capitalize_substitution(matched_abb, sub)
    }
  )
}

#let subs(body) = {
  show "muzero": [MuZero]
  show "azero": [AlphaZero]
  show "effzero": [EfficientZero]
  show "simsiam": [SimSiam]

  show: abbrev("rl", "reinforcement learning")
  show: abbrev("mcts", "monte carlo tree search")
  show: abbrev_plural("impl", "implementation")
  show: abbrev_plural("algo", "algorithm")
  show: abbrev_plural("arch", "architecture")
  show: abbrev_plural("pred", "prediction")
  show: abbrev_plural("repr", "representation")
  show: abbrev_plural("latrep", "latent repr")

  show: abbrev_plural("net", "network")
  show: abbrev("rnet", "representation net")
  show: abbrev("pnet", "prediction net")
  show: abbrev("dnet", "dynamics net")

  show: abbrev("sp", "single player")
  show: abbrev("2p", "two-player")
  show: abbrev("mp", "multiplayer")

  show: abbrev("0sum", "zero-sum")
  show: abbrev_plural("traj", "trajectory")

  show: abbrev_plural("env", "environment")
  show: abbrev_plural("obs", "observation")
  show: abbrev_plural("rew", "reward")
  show: abbrev_plural("val", "value")
  show: abbrev_plural("pol", "policy")
  show: abbrev_plural("ts", "terminal state")
  show: abbrev_plural("tn", "terminal node")
  show: abbrev("vtar", "value target")
  show: abbrev("df", "discount factor")
  show: abbrev("ns", "n-step")
  show: abbrev("nsh", "n-step horizon")
  show: abbrev("offpol", "off-policy")
  show: abbrev("sampeff", "sample efficiency")
  body
}

