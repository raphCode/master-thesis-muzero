#import "@preview/cetz:0.1.2"

#let catch = cetz.canvas({
  import cetz.draw: *

  let (cols, rows) =  (5, 10)
  let block = (1, 2)
  let agent_col = 3

  scale(0.3)
  set-style(stroke: gray)
  for x in range(cols) {
    for y in range(rows) {
      group({
        set-origin((x, -y))
        if (x, y) in (block, (agent_col, rows - 1)) {
          set-style(fill: black)
        }
        rect((0, 0), (1, 1))
      })
    }
  }
})
