#import "@preview/cetz:0.1.1": canvas, draw, tree
#import "util.typ": padding
#import "icon.typ": tic_tac_toe
#import "muzero.typ": net_inference
#import draw: *

#let path_length(node) = {
  calc.max(int(node.path), ..node.children.map(c => path_length(c) + 1))
}

#let on_path(node) = {
  node.path or node.children.any(on_path)
}

#let get_phase(iteration, frame) = {
  calc.max(0, frame - iteration + 1)
}

#let enum_edge = (hide: 0, gray: 1, select: 2, active: 3, forward: 4, backward: 5)
// node example: (draw: 1337, edge: enum_edge.hide, inference: true)

#let build_tree_data(root, iteration, frame) = {
  let final_iteration = iteration >= path_length(root)
  let phase = calc.max(0, frame - iteration + 1)

  let handle_node(node, depth: 0, show_parent: false) = {
    let draw = {
      if on_path(node) and depth < iteration {
        depth
      } else if on_path(node) and depth == iteration and phase > 0 {
        depth
      } else {
        final_iteration
      }
    }
    let edge = {
      let parent_depth = depth - 1
      if on_path(node) and parent_depth < iteration and not final_iteration {
        if phase == 0 {
          if parent_depth < frame {
            enum_edge.active
          } else if parent_depth == frame {
            enum_edge.select
          } else {
            enum_edge.gray
          }
        } else if phase == 1 {
          if depth == iteration {
            enum_edge.forward
          } else {
            enum_edge.active
          }
        } else if phase == 2 {
          enum_edge.active
        } else if phase == 3 {
            enum_edge.backward
        } else {
          enum_edge.gray
        }
      } else if show_parent {
        enum_edge.gray
      } else {
        enum_edge.hide
      }
    }

    let recurse = handle_node.with(depth: depth + 1, show_parent: draw != false)
    let children = node.children.map(recurse)
    let inference = phase == 2 and depth == iteration
    ((draw: draw, edge: edge, inference: inference), ..children)
  }
  handle_node(root)
}

#let flip_anchor(anchor) = {
  if anchor == "left" { return "right" }
  if anchor == "right" { return "left" }
  anchor
}

#let draw_tree(tree_data, alphazero: true, hide_repr_net: false, initial: false) = canvas(length: 1cm, {
  let nodesize = if alphazero { 0.6 } else { 0.35 }
  let arrowdist = if alphazero { 0.9 } else { 0.5 }
  let highlight = blue.lighten(50%)

  let highlight_style(content) = group({
    set-style(
      mark: (stroke: 2pt + highlight, fill: highlight, size: 0.2),
      stroke: 2pt + highlight,
    )
    content
  })

  tree.tree(
    tree_data,
    spread: 1.7,
    grow: if alphazero { 3 } else { 2 },
    draw-node: (node, parent) => {
      anchor("node", ())
      let data = node.content

      // Add playceholder circle for tree sizing
      circle("node", radius: nodesize, stroke: none)

      let highlight_expanded(content) = group({
        if data.edge == enum_edge.forward {
          set-style(stroke: highlight)
        }
        content
      })

      if data.draw == true {
        circle("node", radius: nodesize, stroke: gray)
      } else if data.draw != false {
        let n = data.draw
        if alphazero {
          highlight_expanded(tic_tac_toe("node", n: n + 3))
        } else {
          highlight_expanded(circle("node", radius: nodesize))
          let s = [$s^#n$<no-join>]
          if data.edge == enum_edge.forward {
            s = text(highlight, s)
          }
          content("node", s)
        }
        get-ctx(ctx => {
          let flip = false
          if parent != none {
            let (px, py, _) = coordinate.resolve(ctx, parent)
            let (nx, ny, _) = coordinate.resolve(ctx, "node")
            if px > nx {
              flip = true
            }
          }
          let maybe_flip(anchor) = if flip { flip_anchor(anchor) } else { anchor }
          let offset(dx) = (rel: (x: dx), to: "node")

          if data.inference {
            highlight_style(net_inference(offset(arrowdist), offset(1 + arrowdist), "pred"))
            content(offset(1.1 + arrowdist), text(highlight, strong($ v, p $)), anchor: "left")
          }
          let v = content(
            offset(-(arrowdist + (0.1 * int(not alphazero)))),
            $ overline(v) $,
            anchor: "right",
            name: "v"
          )
          if not initial and data.edge != enum_edge.forward {
            v
          }
          if data.edge == enum_edge.backward or data.inference {
            circle("v", radius: 0.2, stroke: none, fill: highlight)
            v
          }

          if parent != none {
            let b = (a: "node", number: arrowdist + 0.15 + 0.3, abs: true, b: parent)
            if not alphazero and data.edge in (enum_edge.forward, enum_edge.backward) {
              content(
                b,
                text(highlight, $ r^#n $),
                anchor: "top-" + maybe_flip("right"),
                padding: 0.05
              )
            }

            if data.edge == enum_edge.forward {
              content(
                (a: parent, b: "node", number: arrowdist + 0.3, abs: true),
                text(highlight, strong($ a $)),
                anchor: "bottom-" + maybe_flip("left"),
                padding: 0.05,
              )
            }
          }
        })
      }
    },
    draw-edge: (from, to, node) => {
      let a = (a: from, number: arrowdist, abs: true, b: to)
      let b = (a: to, number: arrowdist, abs: true, b: from)
      let draw_edge = node.content.edge
      if draw_edge == enum_edge.gray {
        line(a, b, stroke: gray)
      } else if draw_edge == enum_edge.select {
        highlight_style(line(a, b))
      } else if draw_edge == enum_edge.active {
        line(a, b)
      } else if draw_edge == enum_edge.forward {
        if alphazero {
          highlight_style(line(a, b, mark: (end: ">")))
        } else {
          highlight_style(net_inference(a, b, "dyn"))
        }
      } else if draw_edge == enum_edge.backward {
        highlight_style(line(b, a, mark: (end: ">")))
      }
    },
    name: "tree",
  )

  if alphazero {
    return
  }
  padding(
    tic_tac_toe((rel: (y: 3), to: "tree.0")),
    name: "obs",
    amount: 0.2
  )
  content("obs.top", $ s $, anchor: "bottom")
  if not hide_repr_net {
    let repr_inference = net_inference(
      "obs.bottom",
      (rel: (y: arrowdist), to: "tree.0"),
      "repr"
    )
    let root = tree_data.at(0)
    if root.edge == enum_edge.forward {
      repr_inference = highlight_style(repr_inference)
    }
    repr_inference
  }
})
