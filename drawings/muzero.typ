#import "@preview/cetz:0.1.1": canvas, draw, tree, coordinate
#import "util.typ": bez90, padding, export_anchors, bez_vert
#import "icon.typ": tic_tac_toe, minitree
#import draw: *

#let (rep, dyn, pred) = ($h$, $g$, $f$)

#let loss_arrow_style(content) = group({
  let s = gray + 3pt
  set-style(
    mark: (end: ">", stroke: s, size: 0.2, fill: gray),
    stroke: s,
  )
  content
})

#let net_inference(from, to, net, name: none) = {
  line(from, to, mark: (end: ">"))
  let arrowsize = 0.15
  let plain_end = (a: to, b: from, number: arrowsize, abs: true)
  let mid = (from, 0.5, plain_end)

  group({
    set-style(
      fill: gray,
      stroke: none,
    )
    anchor("end", plain_end)
    let size = 0.2
    let (symbol, txt) = if net == "repr" {
      let s = 1.7 * size
      (
        {
          get-ctx(ctx => {
            let (x, y, _) = coordinate.resolve(ctx, to)
            rotate(calc.atan2(x, -y))
          })
          line((0deg, s), (120deg, s), (240deg, s))
        },
        rep
      )
    } else if net == "pred" {
      let s = 1 * size
      (rect((-s, -s), (s, s)), pred)
    } else {
      (circle((0, 0), radius: size), dyn)
    }
    group({
      set-origin(mid)
      symbol
    })
    content(mid, $ txt $)
  }, name: name)
}

#let training = canvas(length: 1cm, {
  let nodesize = 0.35
  let arrowdist = 0.5

  let col(n) = "col_" + str(n)

  let states = (3, 4, 5, 9)

  let t(n) = if n == 0 [ $t$ ] else [ $t + #n$ ]

  let draw_column(n) = group({
    let last = n == states.len() - 1
    let first = n == 0
    padding(
      tic_tac_toe((4 * n, 0), n: states.at(n)),
      name: "obs",
      amount: 0.2
    )
    export_anchors("obs")
    content(
      "obs.top",
      $ s_#if last [T] else {t(n)} $,
      anchor: "bottom",
    )
    if not last {
      padding(
        circle((rel: (y: -3), to: "obs"), radius: nodesize),
        name: "node",
        amount: arrowdist - nodesize,
      )
      export_anchors("node")
      content(
        "node",
        $ s_t^#n $,
      )
      content(
        (rel: (y: -2), to: "node"),
        $ v_t^#n $,
        anchor: "top",
        name: "value",
        padding: 0.2
      )
      content(
        (rel: (y: 0.2), to: "value.bottom"),
        $ p_t^#n $,
        anchor: "top",
        name: "policy_pred",
        padding: 0.2,
      )
      net_inference(
        "node.bottom",
        "value.top",
        "pred",
      )
      export_anchors("value")
      minitree((rel: (y: -2.5), to: "policy_pred"), name: "tree")
      content("tree.top", $ pi_#t(n) $, anchor: "bottom", padding: 0.1, name: "policy_tree")
      loss_arrow_style(line("policy_tree.top", "policy_pred.bottom"))
    }
    if first {
      net_inference(
        "obs.bottom",
        "node.top",
        "repr",
      )
    }
  }, name: "col_" + str(n))

  let obs_from_to(n) = (col(n - 1) + ".obs-right", col(n) + ".obs-left")

  let action_and_reward(n, reward) = {
    let (from, to) = obs_from_to(n)
    content(
      from,
      $ a_#t(n - 1) $,
      anchor: "top-left",
    )
    content(
      (a: to, b: from, number: 0.15, abs: true),
      reward,
      anchor: "top-right",
      name: "reward_game",
    )
  }

  let value_loss_offset(pos) = (rel: (x: 0.1), to: pos)

  let connect_prev(n) = group({
    assert(n > 0)
    assert(n < states.len() - 1)
    let (from, to) = obs_from_to(n)
    line(from, to, mark: (end: ">"))
    set-style(content: (padding: 0.1))
    action_and_reward(n, $ r_#t(n) $)
    net_inference(
      col(n - 1) + ".node-right",
      col(n) + ".node-left",
      "dyn",
      name: "dyn",
    )
    content(
      "dyn.end",
      $ r_t^#n $,
      anchor: "bottom-right",
      name: "reward_dyn",
    )
    content(
      col(n - 1) + ".node-right",
      $ a_#t(n - 1) $,
      anchor: "bottom-left",
    )
    loss_arrow_style({
      bez_vert(
        "reward_game.bottom",
        (rel: (y: 0.15), to: "reward_dyn.top"),
        x: 1
      )
      line(col(n) + ".value-left", value_loss_offset(col(n - 1) + ".value-right"))
    })
  }, name: "conn_" + str(n))

  let connect_last() = group({
    let n = states.len() - 1
    let (from, to) = obs_from_to(n)
    set-style(content: (padding: 0.1))
    line(from, to, mark: (end: ">"), stroke: (dash: "dashed"))
    let p = 0.2
    line(from, (from, p, to))
    line((to, p, from), to)
    action_and_reward(n, $ z $)
    loss_arrow_style(bez90(
      "reward_game.bottom",
      value_loss_offset("col_" + str(n - 1) + ".value-right"),
    ))
  })

  for (n, _) in states.enumerate() {
    draw_column(n)
    if n > 0 and n < states.len() - 1{
      connect_prev(n)
    }
  }
  connect_last()
})

#let mcts = canvas(length: 1cm, {
  let data = (
    0,
    (
      1,
      (true, true, false),
      (3, 4, true),
    ),
    (
      true,
      (true, true, true, true),
    ),
  )

  let nodesize = 0.35
  let arrowdist = 0.5

  tree.tree(
    data,
    spread: 2,
    grow: 3,
    draw-node: (node, parent) => {
      anchor("node", ())
      if node.content == true {
        circle("node", radius: nodesize, stroke: gray)
      } else if node.content != false {
        circle("node", radius: nodesize)
        let n = node.content
        content("node", [$s^#n$<no-join>], name: "node")
        if parent != none {
          get-ctx(ctx => {
            let (px, py, _) = coordinate.resolve(ctx, parent)
            let (nx, ny, _) = coordinate.resolve(ctx, "node")
            let (anchors, m) = if px < node.x {
              (("left", "right"), -1)
            } else {
              (("right", "left"), 1)
            }
            let (a_pred, a_dyn) = anchors
            let offset(dx, to: "node") = (rel: (x: dx * m), to: to)

            let b = (a: "node", number: arrowdist + 0.15 + 0.3, abs: true, b: parent)
            content(
              b,
              $ r^#n $,
              anchor: "top-" + a_dyn,
              padding: 0.05
            )

            net_inference(offset(-arrowdist), offset(-1.3), "pred")
            content(offset(-1.4), $ v^#n, p^#n $, anchor: a_pred)

            let a = (a: parent, number: arrowdist, abs: true, b: "node")
            content(
              (a: parent, b: "node", number: arrowdist + 0.3, abs: true),
              $ a^#(n - 1) $,
              anchor: "bottom-" + a_pred,
              padding: 0.05,
            )
          })
        }
      }
    },
    draw-edge: (from, to, node) => {
      let a = (a: from, number: arrowdist, abs: true, b: to)
      let b = (a: to, number: arrowdist, abs: true, b: from)
      if node.content == true {
        line(a, b, stroke: gray, mark: (end: ">", stroke: gray))
      } else if node.content != false {
        net_inference(a, b, "dyn")
        let act = (from, 0.2, to)
      }
    },
    name: "tree",
  )

  padding(
    tic_tac_toe((rel: (y: 3), to: "tree.0")),
    name: "obs",
    amount: 0.2
  )
  content("obs.top", $ s $, anchor: "bottom")
  net_inference(
    "obs.bottom",
    (rel: (y: arrowdist), to: "tree.0"),
    "repr"
  )
})
