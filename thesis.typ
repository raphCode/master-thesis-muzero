#let thesis(
  doc,
  debug_equation_join: false,
) = {
  let header_ascent = 30%
  set page(
    margin: (x: 12.5%, y: 8.7%),
    numbering: "1/1",
    header-ascent: header_ascent,
    header: locate(loc => {
      layout(header_size => {
        set text(10pt)

        let hdg_slctr(lvl) = heading.where(level: lvl)

        let prev_heading(lvl) = {
          if lvl < 1 {
            return
          }
          let slctr = hdg_slctr(lvl).before(loc)
          let parent_heading = prev_heading(lvl - 1)
          if parent_heading != none {
            slctr = slctr.after(parent_heading.location())
          }
          query(slctr, loc).at(-1, default: none)
        }

        let next_heading(lvl) = {
          if lvl < 1 {
            return
          }
          let slctr = hdg_slctr(lvl).after(loc)
          let parent_heading = next_heading(lvl - 1)
          if parent_heading != none {
            slctr = slctr.before(parent_heading.location())
          }
          query(slctr, loc).at(0, default: none)
        }

        let current_page = counter(page).at(loc).first()
        let page_body_start_y = header_size.height / (1 - float(header_ascent))

        let current_heading(lvl) = {
          // get the previous heading if the next one does not start directly at the top
          // of the current page
          let next_hdg = next_heading(lvl)
          if next_hdg != none {
            let next_loc = next_hdg.location().position()
            if next_loc.page == current_page and next_loc.y <= page_body_start_y + 10pt {
              return
            }
          }
          prev_heading(lvl)
        }

        let format_hdg(lvl, formatter) = {
          let hdg = current_heading(lvl)
          if hdg != none {
            formatter(hdg)
          }
        }

        format_hdg(1, hdg => {
          smallcaps(hdg.body)
          format_hdg(2, hdg => {
            let next_lvl_hdg = current_heading(3)
            if next_lvl_hdg == none {
              h(1fr)
              hdg.body
            } else {
              " - "
              hdg.body
              h(1fr)
              emph(next_lvl_hdg.body)
            }
          })
        })
      })
    })
  )
  set text(
    font: "New Computer Modern",
    size: 12pt
  )
  set par(
    justify: true,
  )
  set figure(
    gap: 1.5em,
  )
  show figure: fig => block(
    fig,
    width: 100%,
    spacing: 2em,
  )
  set heading(numbering: "I - A 1.1:")
  show heading.where(level: 1): hd => pagebreak(weak: true) + hd

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

#let neq(it) = {
  set math.equation(numbering: "(1)")
  it
}
