#let thesis(
  doc,
  debug_equation_join: false,
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

  set cite(brackets: true, style: "chicago-author-date")
  show cite.where(brackets: true): it => "[" + cite(..it.keys, brackets: false)  + "]"

  show math.equation.where(block: false): eq => {
    let join_nonbreaking = h(0pt, weak: true) + [~] + h(0pt, weak: true)
    let eq_label = eq.at("label", default: none)
    let allow_join_left = eq_label not in (<no-join>, <join-right>)
    style(sty => {
      let size = measure(eq, sty)
      let threshold = measure(h(3em), sty)
      if size.width < threshold.width and allow_join_left {
        if debug_equation_join {
          join_nonbreaking
          [$<-$<no-join>]
        }
        join_nonbreaking
      }
    })
    eq
    if eq_label == <join-right> {
      join_nonbreaking
      if debug_equation_join {
        [$->$<no-join>]
        join_nonbreaking
      }
    }
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
