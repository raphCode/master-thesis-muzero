#import "@preview/cetz:0.1.1": canvas, draw, tree
#import "util.typ": padding, bez90
#import "icon.typ": tic_tac_toe, database
#import "muzero.typ": net_inference
#import "stepwise_mcts.typ"

#let architecture_overview(show_mcts) = canvas(length: 1cm, {
  import draw: *

  let pad = 0.3

  set-style(fill: rgb(0%, 0%, 50%, 15%))

  let bbox(label, inner, hidden: false) = {
    padding(
      {
        inner
        if hidden {
          label = hide(label)
        }
        content((rel: (y: 0.5), to: "box.top"), label, anchor: "bottom")
      },
      name: "cnt",
      amount: pad,
    )
    if hidden {
      set-style(fill: none, stroke: none)
    }
    rect("cnt.bottom-left", "cnt.top-right", name: "box")
  }

  bbox([Selfplay],
    bbox([MCTS],
      content(
        (0, 0),
        [Neural Network],
        padding: pad,
        frame: "rect",
        anchor: "left",
        name: "box",
      ),
    hidden: not show_mcts),
  )

  set-style(fill: none)
  database((-5, 0), size: 0.6, name: "data")

  set-style(stroke: 2pt)
  let arrow_mark = (end: ">", fill: black, size: 0.2)
  bez90(
    (rel: (y: -0.5), to: "box.top-left"),
    (rel: (y: -0.6), to: "data.top"),
    flip: true,
    mark: arrow_mark,
    name: "gen",
  )
  line(
    (vertical: (0, 0), horizontal: "data.right"),
    (0,0),
    mark: arrow_mark,
    name: "train"
  )

  content("data.bottom", [Training Data], padding: 0.3, anchor: "top")
  content("gen.top", [Generation], padding: 0.1, anchor: "bottom")
  content("train.top", [Training], padding: 0.1, anchor: "bottom")
})

#let muzero_network_arrow(net, length: 2) = canvas(length: 1cm, {
  net_inference((0, 0), (length, 0), net)
})

#let animate_mcts(tree_def, alphazero: false) = {
  let draw_tree(iteration: 0, frame: 0, ..args) = {
    stepwise_mcts.draw_tree(
      stepwise_mcts.build_tree_data(tree_def, iteration, frame),
      alphazero: alphazero,
      ..args,
    )
  }

  let n = stepwise_mcts.path_length(tree_def)

  if not alphazero {
    [== MCTS: !Obs]
    draw_tree(
      iteration: -1,
      frame: 0,
      hide_repr_net: true,
    )

    [== MCTS: !RNET]
    draw_tree(
      iteration: 0,
      frame: 0,
    )
  }

  [== MCTS: Root !N]
  draw_tree(
    iteration: 0,
    frame: 4,
    initial: true,
  )

  [== MCTS: Root !N: Network Inference]
  draw_tree(
    iteration: 0,
    frame: 1,
  )

  [== MCTS: Root !N]
  draw_tree(
    iteration: 0,
    frame: 4,
  )

  for i in range(1, n) {
    for frame in range(i + 4) {
      let p = stepwise_mcts.get_phase(i, frame)
      let phases = (
        [: Selection],
        [: Expansion],
        [: Expansion (!Net inference)],
        [: Backpropagation],
        [],
      )
      heading(level: 2)[MCTS: Iteration #i#phases.at(p)]
      draw_tree(
        iteration: i,
        frame: frame,
      )
    }
  }

  [== MCTS: After many Iterations]
  draw_tree(
    iteration: n,
    frame: 0,
  )
}

