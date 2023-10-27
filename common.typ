#let exret_formula = $sum_(k=1)^(T-t) gamma^(k-1) r_(t+k)$

#let masked_prior_from_pred(
  pred: a => $p(#a |s)$,
  state: $s$,
) = {
  $ P(state, a) = cases(
    m pred(a) &"if" a in A(state) \
    0 &"if" a in.not A(state) \
  ) \
  "with" m = frac(1, sum_(b in A(state)) pred(b)) $
}
