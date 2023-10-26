#import "@preview/cetz:0.1.2": canvas, tree, draw

#let rl_transition = canvas(length: 1cm, {
  import draw: *

  let node(pos, label, name: none) = {
    circle(pos, radius: 0.5, name: name)
    content(pos, label)
  }
  
  let connect(from, to, name: none) = {
    let dist = 0.7 + 0.15
    line(
      (a: from, b: to, number: dist, abs: true),
      (a: to, b: from, number: dist, abs: true),
      mark: (end: ">"),
      name: "line",
    )
  }

  node((0, 0), $ s_t $, name: "s1")
  node((5, 0), $ s_(t+1) $, name: "s2")
  connect("s1", "s2", name: "line")
  
  content("line.left", $ a_t $, anchor: "bottom-left", padding: 0.15)
  content("line.right", $ r_(t+1) $, anchor: "bottom-right", padding: 0.15)
})
