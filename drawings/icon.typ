#import "@preview/cetz:0.1.1": draw, tree
#import "util.typ": padding, pairwise
#import draw: *

#let minitree(pos, name: none) = group({
  set-origin(pos)
  let data = (0, (0, 0, 0), (0, 0, 0))
  let nodesize = 0.1
  set-origin((nodesize, -nodesize))
  //set-style(stroke: 0.5pt)
  tree.tree(
    data,
    spread: 0.3,
    grow: 0.4,
    draw-node: (_, _) => circle((), radius: nodesize),
    draw-edge: (from, to, _) => {
      line(
        (a: from, number: nodesize, abs: true, b: to),
        (a: to, number: nodesize, abs: true, b: from),
      )
    }, name: "tree")
    anchor("center", "tree")
}, anchor: "bottom-right", name: name)

#let tic_tac_toe(pos, n: 5, name: none) = group({
  set-origin(pos)
  scale(0.4)
  let size = 3
  set-origin((-size / 2, -size / 2))
  for i in range(1, size) {
    line((0, i), (size, i))
    line((i, 0), (i, size))
  }
  let moves = (
    (0, 0),
    (1, 1),
    (1, 2),
    (0, 2),
    (2, 0),
    (1, 0),
    (2, 2),
    (2, 1),
    (0, 1),
  )
  for (move, coord) in moves.slice(0, n).enumerate() {
    let pos = (rel: (0.5, 0.5), to: coord)
    let r = 0.3
    if calc.even(move) {
      circle(pos, radius: r)
    } else {
      line((rel: (-r, -r), to: pos), (rel: (r, r), to : pos))
      line((rel: (-r, r), to: pos), (rel: (r, -r), to : pos))
    }
  }
}, name: name)

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

#let dice(pos, name: none) = group({
  set-origin(pos)

  let side(x, y, number: none) = group({
    rotate((x: -90deg * x, y: 90deg * y))
    set-origin((0, 0, -1))
    rect((-1, -1), (1, 1), stroke: 0.5pt)
    set-style(fill: black, radius: 0.1)
    circle((0, 0))
    for i in range(4) {
      if number == 5 or calc.rem(i, 2) == 0 and number == 3 {
        circle((45deg + i * 90deg, 0.7))
      }
    }
  })

  scale(0.3)

  side(1, 1, number: 1)
  side(0, 1, number: 3)
  side(0, 0, number: 5)
}, name: name)

#let database(pos, size: 0.3, name: "") = padding({
  set-origin(pos)
  scale(size)

  set-style(stroke: 0.7pt)
  scale((y: 0.5))
  let n = 3
  let r = 2
  set-origin((0, -n / 2))
  for y in range(n) {
    arc((r, y), start: 0deg, stop: -180deg, radius: r)
  }
  circle((0, n), radius: r)
  line((r, 0), (r, n))
  line((-r, 0), (-r, n))
}, name: name)
