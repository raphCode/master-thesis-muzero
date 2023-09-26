#import "@preview/cetz:0.1.1": canvas, draw
#import draw: *

#let pairwise(vec) = {
  vec.zip(vec.slice(1))
}

#let padding(content, name: none, anchor: none, amount: 0.1) = {
  let a(n, x, y) = draw.anchor(n, (rel: (x * amount, y * amount), to: "content." + n))
  group({
    group(content, name: "content")
    a("left", -1, 0)
    a("right", 1, 0)
    a("bottom", 0, -1)
    a("top", 0, 1)
    a("bottom-left", -1, -1)
    a("bottom-right", 1, -1)
    a("top-left", -1, 1)
    a("top-right", 1, 1)
  }, name: name, anchor: anchor)
}

#let bez_vert(a, b, x: 2) = {
  let c = (rel: (0, -x), to: a)
  let d = (rel: (0, x), to: b)
  bezier(a, b, c, d, mark: (end: ">"))
}

#let bez_hor(a, b, x: 2, name: none) = {
  let c = (rel: (x, 0), to: a)
  let d = (rel: (-x, 0), to: b)
  bezier(a, b, c, d, mark: (end: ">"), name: name)
}
