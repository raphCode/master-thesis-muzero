#import "@preview/cetz:0.1.1": canvas, draw, tree

#let data = (
  auto, ([], [], []), (auto, (auto, [], none, []), [])
)

#let draw_phase(phase) = {
  assert(1 <= phase and phase <= 4)

  canvas(length: 0.7cm, {
    import draw: *

    let node-radius = 0.3

    set-style(content: (padding: .2),
      stroke: black,
      circle: (radius: node-radius),
      mark: (fill: black),
    )

    let draw_edge(from, to, node) = {
      if phase in (1, 4) and node.content in (auto, none) {
        if phase == 4 {
          (from, to) = (to, from)
        }
        line(
          (a: from, number: node-radius, abs: true, b: to),
          (a: to, number: node-radius + 0.2, abs: true, b: from),
          stroke: 2pt,
          mark: (stroke: 2pt, end: ">"),
        )
      } else {
        line(
          (a: from, number: node-radius, abs: true, b: to),
          (a: to, number: node-radius, abs: true, b: from),
        )
      }
    }

    tree.tree(
      data,
      spread: 0.9,
      grow: 1.5,
      draw-node: (node, _) => {
        if node.content == auto {
          if phase in (1, 4) {
            circle((), stroke: 2pt)
          } else {
            circle(())
          }
        } else if node.content == none {
          if phase in (2, 4) {
            circle((), stroke: 2pt)
          } else if phase == 3 {
            circle((), name: "node")
            let offset = (0, -2.5)
            rect(
              (rel: offset, to: "node.bottom-left"),
              (rel: offset, to: "node.top-right"),
              stroke: 2pt,
              name: "terminal"
            )
            let from = "node.bottom"
            let to = "terminal.top"
            let start = (a: from, number: 0.1, abs: true, b: to)
            let end = (a: to, number: 0.5, abs: true, b: from)
            let n = 6
            let amplitude = .1
            let points = (
              start,
              ..for i in range(n) {
                amplitude *= -1
                ((rel: (amplitude, 0), to: (a: start, number: (i + 0.5) / n , b: end)), )
              },
              end,
              (a: to, number: 0.2, abs: true, b: from),
            )
            line(..points, stroke: 2pt, mark: (end: ">", stroke: 2pt))
          }
        } else {
          circle(())
        }
      },
      draw-edge: draw_edge,
      name: "tree",
    )

    // Draw a custom third child under the root without widening the tree
    circle(("tree.0-0", 0.5, "tree.0-1"), name: "aux-child")
    draw_edge("tree.0", "aux-child", (content: []))
  })
}
