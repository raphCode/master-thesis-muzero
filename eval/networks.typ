#let res(body) = [Residual block consisting of: #body]
#let linear(out_size, act: [ReLU]) = [
  Fully connected layer with #out_size output neurons #h(0pt, weak: true)
  #if act != none [, #act activation] else []
]
#let tensor_shape(shape) = shape.map(str).intersperse[$times$<no-join>].sum()

#let dnet_input(latent_size, action_size) = [
  The #{latent_size}-dimensional latent tensor is concatenated with the
  #{action_size}-element !a onehot to yield a tensor with #{latent_size + action_size}
  elements.
]

#let pnet_input(latent_size) = [
  The #{latent_size}-dimensional latent tensor is processed by the following layers:
]

#let dnet_output(latent_size, support_shape, turn_size) = [
  The resulting #{latent_size + support_shape.product() + turn_size}-dimensional tensor is
  split and reshaped into three tensors, with #latent_size elements, shape
  #tensor_shape(support_shape) and #turn_size elements respectively.
]

#let pnet_output(support_shape, action_size) = [
  The resulting #{support_shape.product() + action_size}-dimensional tensor is split and
  reshaped into two tensors, with shape #tensor_shape(support_shape) and #action_size
  elements respectively.
]
