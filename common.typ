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

#let wc = $w_frak(C)$
#let wt = $w_frak(T)$


#let series(x, start: 0, end: $K$) = {
  let sub(n) = { if n == 0 {} else { $+#n$ } }
  let item(n) = math.attach(x, b: $t$ + sub(n))
  $#item(start), #item(start + 1), ..., #x _#end$
}
#let series_pi = series($pi$)
#let series_p = series($p$)
#let series_a = series($a$, end: $K - 1$)
#let series_r = series($r$, start: 1)
#let series_w = series($w$, start: 1)
#let series_g = series($G$, start: 1)
