#import "@preview/cetz:0.1.1": draw, tree
#import "util.typ": padding, pairwise
#import draw: *

#let network(pos, ct, name: "") = {
  padding({
    set-origin(pos)
    set-style(stroke: 0.7pt)
    let units = (2, 3, 2)
    let x_offset = ((units.len() - 1) / 2)
    let poss = for (i, n) in units.enumerate() {
      let y_offset = (n - 1) / 2
      (for j in range(n) {
        (((i - x_offset) * 0.4, (j - y_offset) * 0.3), )
      }, )
    }
    for (pos_a, pos_b) in pairwise(poss) {
      for a in pos_a {
        for b in pos_b {
          line(a, b)
        }
      }
    }
    for pos in poss {
      for p in pos {
        circle(p, fill: white, radius: 0.1)
      }
    }
  }, name: name)
  content((rel: (y: -0.1), to: name + ".bottom"), align(center, ct), anchor: "top")
}
