#let no-mangle(body) = [#body<no-mangle>]

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
  let join_nonbreaking = h(0pt, weak: true) + [~]
  set cite(brackets: true, style: "chicago-author-date")
  show cite.where(brackets: true): it => "[" + cite(..it.keys, brackets: false)  + "]"
  show math.equation: it => if not it.block and it.at("label", default: none) != <no-mangle> {
    h(0pt, weak: true) + [~] + it
  } else {
    it
  }

  show regex("(?i:on the other hand)"): m => {
    panic("Prof. Andre only allows to use this phrase with 'on one hand' beforehand!")
    /*
    Panic unconditionally, since often the phrase is used in the sense of 'conversely'.
    Sorry if you used the phrase 'properly'.
    Or add some typst locate magic to suppress the false positives.
    */
  }
  doc
}

#let citet(key) = {  // cite in text
  show regex("\d{4}$"): match => "[" + match.text + "]"
  cite(key, brackets: false)
}

#let blockquote(citekey, body) = {
  block(
    fill: luma(230),
    inset: 8pt,
    radius: 4pt,
    {
      ["]
      h(0pt, weak: true)
      body
      h(0pt, weak: true)
      ["]
      align(right, cite(citekey))
    },
  )
}
