#let citet(key) = {  // cite in text
  show regex("\d{4}$"): match => "[" + match.text + "]"
  cite(key, brackets: false, style: "chicago-author-date")
}
