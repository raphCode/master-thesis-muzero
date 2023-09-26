#import "@preview/cetz:0.1.1": canvas, draw
#import "util.typ": padding, bez_vert, bez_hor

#let pairwise(vec) = {
  vec.zip(vec.slice(1))
}

#let sl = $p_sigma$
#let rl = $p_rho$
#let roll = $p_pi$
#let v = $v_theta$

#let training_pipeline = canvas(length:1cm, {
  import draw: *

  let dataset(pos, ct, name: "") = {
    padding({
      set-origin(pos)
      scale(0.3)

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
    content((rel: (y: 0.1), to: name + ".top"), align(center, ct + [\
    Games]), anchor: "bottom")
  }

  let net(pos, ct, name: "") = {
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

  let text(pos, ct, anchor) = content(pos, align(center, emph(ct)), anchor: anchor, padding: 0.2)

  let arrow(from, to, name: none) = {
    line(from, to, mark: (end: ">"), name: name)
  }

  dataset((0, 0), [Human Expert], name: "kgs")

  net(
    (2, -4),
    [SL policy\
    network\
    #sl],
    name: "sl"
  )
  bez_vert("kgs.bottom", "sl.top")
  net(
    (-2, -4),
    [rollout policy\
    network\
    #roll],
    name: "rollout"
  )
  bez_vert("kgs.bottom", "rollout.top")

  text((0, -2.1), [Classification], "top")

  net(
    (7, -4),
    [RL policy\
    network\
    #rl],
    name: "rl"
  )
  arrow("sl.right", "rl.left", name: "pg")
  text("pg.center", [Selfplay and\
  Policy Gradient], "bottom")

  dataset((11, 0), [Selfplay], name: "rldata")
  bez_hor("rl.right", "rldata.left", name: "selfplay")
  text("selfplay.center", [Selfplay], "right")

  net(
    (11, -4),
    [value\
    network\
    #v],
    name: "vnet"
  )
  arrow("rldata.bottom", "vnet.top", name: "regr")
  text("regr.center", [Regression], "left")
})