#import "@preview/cetz:0.1.1": canvas, draw, tree, coordinate
#import "util.typ": bez90, padding, export_anchors, bez_vert, bez_hor
#import "icon.typ": tic_tac_toe, minitree, dice
#import "../common.typ": exret_formula
#import draw: *

#let (rep, dyn, pred) = ($h$, $g$, $f$)

#let vectorize_maybe(use_vectors, cnt) = if use_vectors { $arrow(cnt)$ } else { cnt }

#let loss_arrow_style(content, bidi: false, ..args) = group({
  let s = gray + 3pt
  let mark = (end: ">", stroke: s, size: 0.2, fill: gray)
  if bidi {
    mark.start = ">"
  }
  set-style(
    mark: mark,
    stroke: s,
  )
  content
}, ..args)

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
    let size = 0.3
    let (symbol, txt) = if net == "repr" {
      let s = 1.6 * size
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
      let s = 0.85 * size
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

#let training(
  draw_pnet: true,
  dynamics_env: none,
  dynamics_net: none,
  latent_loss: "",
  chance_event: none,
  use_vectors: false,
  value_target: "outcome",
  loss_label_value: $ell^v$,
  loss_label_policy: $ell^p$,
  loss_label_latent: $ell^l$,
  loss_label_dynamics: $ell^r$,
) = canvas(length: 1cm, {
  let nodesize = 0.45
  let arrowdist = nodesize + 0.15

  let col(n) = "col_" + str(n)

  let states = (3, 4, 5, 9)

  let t(n) = if n == 0 [ $t$ ] else [ $t + #n$ ]

  let vectorize = vectorize_maybe.with(use_vectors)

  if dynamics_env == none {
    if value_target == "return" {
      dynamics_env = n => { (if n == -1 { $ r_T $ } else { $ r_(t+#n) $ }, ) }
    } else if value_target == "outcome" {
      dynamics_env = n => { (if n == -1 { $ z $ } else [], ) }
    }
  }
  if dynamics_net == none {
    if value_target == "return" {
      dynamics_net = n => ($ r_t^#n $, )
    } else if value_target == "outcome" {
      dynamics_net = n => ()
    }
  }

  let stack_dynamics(fn, n) = stack(..fn(n), spacing: 2pt)
  let stack_env = stack_dynamics.with(dynamics_env)
  let stack_net = stack_dynamics.with(dynamics_net)

  let node(dy, label, name: none) = {
    padding(
      circle((rel: (y: -dy), to: "obs"), radius: nodesize),
      name: name,
      amount: arrowdist - nodesize,
    )
    content(name, label)
  }

  let draw_column(n) = group({
    let last = n == states.len() - 1
    let first = n == 0
    let col_offset = 4
    padding(
      if n == chance_event {
        dice((n * col_offset, 0))
      } else {
        tic_tac_toe((n * col_offset, 0), n: states.at(n))
      },
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
      get-ctx(ctx => {
        let (_, h1) = measure(stack_env(n), ctx)
        let (_, h2) = measure(stack_net(n), ctx)
        let dist = if latent_loss != "" { 6 } else { calc.max(h1 + h2 + 2.5, 3) }
        node(dist, $ s_t^#n $, name: "node")
      })
      export_anchors("node")
      if draw_pnet {
        content(
          (rel: (y: -2.5), to: "node"),
          $ vectorize(v)_t^#n $,
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
        if value_target == "outcome" {
          anchor("value_target", (rel: (1.6, 0.5), to: "value.top"))
          loss_arrow_style(bez_hor("value_target", "value.right", x: -1), name: "loss_arrow_value")
          content("loss_arrow_value", $ #loss_label_value $, anchor: "left", padding: 0.2)
        } else if value_target == "return" {
          content(
            (rel: (x: col_offset / 2), to: "value"),
            $ vectorize(G)_#t(n) $,
            padding: 0.1,
            name: "return",
          )
          loss_arrow_style(line("return.left", "value.right", name: "loss_value"), name: "loss_arrow_value")
          content("loss_arrow_value", $ #loss_label_value $, anchor: "bottom", padding: 0.2)
          export_anchors("return")
        }
        let policy
        if n == chance_event {
          dice((rel: (y: -2.5), to: "policy_pred"), name: "policy_icon")
          policy = $c$
        } else {
          minitree((rel: (y: -2.5), to: "policy_pred"), name: "policy_icon")
          policy = $pi$
        }
        content("policy_icon.top", $ #policy _#t(n) $, anchor: "bottom", padding: 0.1, name: "policy_target")
        loss_arrow_style(line("policy_target.top", "policy_pred.bottom"), name: "loss_arrow_policy")
        content("loss_arrow_policy", $ #loss_label_policy $, anchor: "left", padding: 0.2)
      }
    }
    if first {
      net_inference(
        "obs.bottom",
        "node.top",
        "repr",
      )
    } else if latent_loss != "" and not last {
      node(3, $ s_(t+#n)^0 $, name: "node2")
      net_inference(
        "obs.bottom",
        "node2.top",
        "repr",
      )
      let bidi = latent_loss == "bidirectional"
      loss_arrow_style(
        line(
          (rel: (y: int(bidi) * -0.15), to: "node2.bottom"),
          (rel: (y: arrowdist + 0.15), to: "node")
        ),
        name: "loss_arrow_latent",
        bidi: bidi,
      )
      content("loss_arrow_latent", $ #loss_label_latent $, anchor: "left", padding: 0.2)
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

  let connect_prev(n) = group({
    assert(n > 0)
    assert(n < states.len() - 1)
    let (from, to) = obs_from_to(n)
    line(from, to, mark: (end: ">"))
    set-style(content: (padding: 0.1))
    action_and_reward(n, stack_env(n))
    net_inference(
      col(n - 1) + ".node-right",
      col(n) + ".node-left",
      "dyn",
      name: "dyn",
    )
    content(
      "dyn.end",
      stack_net(n),
      anchor: "bottom-right",
      name: "reward_dyn",
    )
    content(
      col(n - 1) + ".node-right",
      $ a_#t(n - 1) $,
      anchor: "bottom-left",
    )
    if draw_pnet and value_target == "outcome" {
      loss_arrow_style(line(
        col(n - 1) + ".value_target",
        col(n) + ".value_target",
        mark: (end: none),
      ))
    }
    if dynamics_net(n).len() > 0 {
      loss_arrow_style(bez_vert(
        "reward_game.bottom",
        (rel: (y: 0.15), to: "reward_dyn.top"),
        x: if latent_loss != "" {2} else {1},
      ), name: "loss_arrow_dynamics")
      content("loss_arrow_dynamics", $ #loss_label_dynamics $, anchor: "right", padding: 0.2)
    }
  }, name: "conn_" + str(n))

  let connect_last() = group({
    let n = states.len() - 1
    let (from, to) = obs_from_to(n)
    set-style(content: (padding: 0.1))
    line(from, to, mark: (end: ">"), stroke: (dash: "dashed"))
    let p = 0.2
    line(from, (from, p, to))
    line((to, p, from), to)
    action_and_reward(n, stack_env(-1)) 
    if draw_pnet {
      if value_target == "outcome" {
        loss_arrow_style(bez90(
          "reward_game.bottom",
          col(n - 1) + ".value_target",
          mark: (end: none),
        ))
      } else if value_target == "return" {
        content(
          (rel: (0, 1), to: col(n - 1) + ".return-top-left"),
          $ G_t = #exret_formula $,
          anchor: "bottom-left",
          frame: "rect",
          padding: 0.1,
        )
      }
    }
  })

  for (n, _) in states.enumerate() {
    draw_column(n)
    if n > 0 and n < states.len() - 1{
      connect_prev(n)
    }
  }
  connect_last()
})

#let mcts(data) = canvas(length: 1cm, {
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
        get-ctx(ctx => {
          let (anchors, m) = (("left", "right"), -1)
          if parent != none {
            let (px, py, _) = coordinate.resolve(ctx, parent)
            let (nx, ny, _) = coordinate.resolve(ctx, "node")
            if px > nx {
              (anchors, m) = (("right", "left"), 1)
            }
          }
          let (a_pred, a_dyn) = anchors
          let offset(dx, to: "node") = (rel: (x: dx * m), to: to)

          net_inference(offset(-arrowdist), offset(-1.5), "pred")
          content(offset(-1.6), $ v^#n, p^#n $, anchor: a_pred)

          if parent != none {
            let b = (a: "node", number: arrowdist + 0.15 + 0.3, abs: true, b: parent)
            content(
              b,
              $ r^#n $,
              anchor: "top-" + a_dyn,
              padding: 0.05
            )

            content(
              (a: parent, b: "node", number: arrowdist + 0.3, abs: true),
              $ a^#(n - 1) $,
              anchor: "bottom-" + a_pred,
              padding: 0.05,
            )
          }
        })
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
