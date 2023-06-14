#let thesis(
  doc,
) = {
  set page(
    margin: (x: 12.5%, y: 8.7%),
  )
  set text(
    font: "New Computer Modern",
    size: 12pt
  )
  set par(
    justify: true,
  )
  set heading(numbering: "1.")
  doc
}

#let citet(key) = {  // cite in text
  show regex("\d{4}$"): match => "[" + match.text + "]"
  cite(key, brackets: false, style: "chicago-author-date")
}
