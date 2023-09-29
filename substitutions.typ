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

#let get_plural_form(text) = {
  if text.match(regex("[^aeiou]y$")) != none {
    text.replace(regex("y$"), "ies")
  } else if text.match(regex("(?:ss|ch|sh|x)$")) != none {
    text + "es"
  } else {
    text + "s"
  }
}

#let showrule_regex_captures(pattern, callback) = body => {
  // apply a show rule to the regex,
  // match it again to get capture groups and call the callback with them
  let r = regex(pattern)
  show r: it => { callback(..it.text.match(r).captures) }
  body
}

#let get_substitution(key, plural_form: false) = {
  let data = yaml("abbrev.yaml").at(key)
  if type(data) == "dictionary" {
    if plural_form { data.plural } else { data.singular }
  } else {
    if plural_form { get_plural_form(data) } else { data }
  }
}

#let subs(body) = {
  let abbrev_dict = yaml("abbrev.yaml")
  for (key, value) in abbrev_dict {
    if key + "s" in abbrev_dict.keys() {
      panic(
        "Plural key of '" + get_substitution(key) +
        "' clashes with key of '" + get_substitution(key + "s") + "'!"
      )
    }
  }

  show: showrule_regex_captures(
    "!(\w+?)(s?)\b",
    (matched_key, plural_s) => {
      let key = lower(matched_key)
      let sub
      if key + "s" in abbrev_dict.keys() {
        sub = get_substitution(key + "s")
      } else if key in abbrev_dict.keys() {
        sub = get_substitution(key, plural_form: plural_s == "s")
      } else {
        panic("Unknown key: !" + key + plural_s)
      }
      capitalize_substitution(matched_key, sub)
    }
  )

  show regex("\bmaxn\b"): [max#super[n]]

  body
}
