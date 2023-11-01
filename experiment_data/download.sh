tags=(
  "loss/latent"
  "loss/turn"
  "loss/policy"
  "loss/reward"
  "loss/value"
  "selfplay/score00"
  "prediction error/reward mse"
  "prediction error/value mse"
)

for run in "$@"
do
  echo "DOWNLOAD $run"
  mkdir -p "$run"
  for tag in "${tags[@]}"
  do
    curl "http://localhost:6009/data/plugin/scalars/scalars?tag=${tag// /+}&run=${run// /+}&format=csv" > "$run/${tag//\// }.csv"
  done
done
