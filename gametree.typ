#import "@preview/cetz:0.1.0": canvas, draw, tree, coordinate, vector

#let backprop(node, parent_player: none) = {
  node.parent_player = parent_player
  if "utility" in node {
    node
  } else {
    node.children = node.children.map(backprop.with(parent_player: node.player))
    let max_utility
    let max_child_index
    for (n, child) in node.children.enumerate() {
      let u = child.utility.at(node.player - 1)
      if max_utility == none or u > max_utility {
        max_utility = u
        max_child_index = n
      }
    }
    let max_child = node.children.at(max_child_index)
    max_child.backprop = true
    node.children.at(max_child_index) = max_child
    node.utility = max_child.utility
    node.utility.at(node.player - 1) = max_utility
    node.backprop_info = (utility: max_utility, action: max_child.label, child: max_child)
    node
  }
}

#let get_optimal_strategy(node) = {
  if "backprop_info" in node {
    return (node.backprop_info.action, ) + get_optimal_strategy(node.backprop_info.child)
  }
  ()
}

#let n(label, data, ..children) = {
  if children.pos().len() == 0 {
    assert(
      type(data) in ("float", "integer", "array"),
      message: "Terminal nodes must specify utility",
    )
    if type(data) == "array" {
      data.map(type).map(t => assert(t in ("float", "integer")))
      return (label: label, utility: data)
    }
    // Assume single player
    (label: label, utility: (data,))
  } else {
    if type(data) == "integer" {
      return (label: label, content: [#data], player: data, children: children.pos())
    }
    assert(type(data) in ("content"), message: "Invalid node content!")
    // Assume single player
    (label: label, content: data, player: 1, children: children.pos())
  }
}

#let l = n.with($a$)
#let r = n.with($b$)

#let nodetree(root_content, ..nodes, backpropagate: false) = {
  let root = n([], root_content, ..nodes)
  if backpropagate {
    root = backprop(root)
  }
  root
}

#let draw_gametree(nodetree) = {
  let make_cetz_tree_data(node) = {
    if "children" not in node {
      return node
    }
    let children = node.remove("children").map(make_cetz_tree_data)
    (node, ..children)
  }

  let data = make_cetz_tree_data(nodetree)

  canvas(length: 1cm, {
    import draw: *

    set-style(
      content: (padding: .2),
      stroke: black
    )

    tree.tree(data, spread: 2, grow: 1.7, draw-node: (node, _) => {
      let data = node.content
      let util_cnt
      if "utility" in data {
        let util = data.utility
        let bp = data.at("backprop", default: false)
        util_cnt = if type(util) == "array" {
          if util.len() == 1 {
            util.at(0)
          } else {
            [(]
            for (n, u) in util.enumerate() {
              if n + 1 == data.at("parent_player", default: none) {
                if bp {
                  (strong(underline([#u])), )
                } else {
                  (underline([#u]), )
                }
              } else {
                ([#u], )
              }
            }.join(", ")
            [)]
          }
        } else {
          util
        }
        if type(util_cnt) != "content" {
          util_cnt = [#repr(util_cnt)]
          if bp {
            util_cnt = text(weight: "bold", util_cnt)
          }
        }
      }
      if "content" in data {
        content((), data.content)
        circle((), radius: .3, stroke: black, name: "c")
        content("c.right", util_cnt, anchor: "left")
      } else {
        // terminal node
        content((), util_cnt)
      }
    }, draw-edge: (from, to, node) => {
      let is_terminal = node.children.len() == 0
      let a = (a: from, number: .5, abs: true, b: to)
      let b = (a: to, number: if is_terminal {.3} else {.5}, abs: true, b: from)
      if node.content.at("backprop", default: false) {
        line(b, a, stroke: 2pt, mark: (end: ">", stroke: 2pt, fill: black))
      } else {
        line(a, b)
      }
      let l = (a: from, number: 0.5, b: to)
      group(ctx => {
        let f = (a, b) => {
          let d = vector.dot(vector.sub(a, b), (1, 0, 0))
          return d < 0
        }
        let left = f(
          coordinate.resolve(ctx, a),
          coordinate.resolve(ctx, b),
        )
        let anchor = if left { "left" } else { "right" }
        content(l, node.content.label, padding: 0.1, anchor: "bottom-" + anchor)
      })
    }, name: "tree")
  })
}
