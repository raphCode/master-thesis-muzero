#import "@preview/cetz:0.1.1": canvas, tree, draw

#let size = 5mm

#let icon_af = rotate(rect(width: size * 2, height: size * 2, radius: 50%), 45deg)
#let icon_s = circle(radius: size)

#let afterstates = canvas(length: 1cm, {
  import draw: *

  let node(pos, label, icon: icon_s, name: none) = {
    content(pos, icon, name: name)
    content(pos, label)
  }
  
  let connect(from, to, label: none) = {
    let dist(name) = 0.7 + 0.15 * int(name.starts-with("af"))
    line(
      (a: from, b: to, number: dist(from), abs: true),
      (a: to, b: from, number: dist(to), abs: true),
      mark: (end: ">"),
      name: "line",
    )
    content("line", label, anchor: "bottom", padding: 0.15)
  }

  node((0, 0), $ s_t $, name: "s")
  
  node((3, 0), $ a s_t $, icon: icon_af, name: "af")
  connect("s", "af", label: $ a_t $)

  let dy = 1.8
  for i in range(3) {
    node((6, (i - 1) * dy), $ s_(t+1)^#i $, name: "st")
    connect("af", "st", label: $ c_t^#i $)
  }
})
