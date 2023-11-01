import sys
import polars
from matplotlib import pyplot as plt

file = sys.argv[1]

raw = polars.read_csv(file)
steps = raw["Step"]

w = 5

v = raw["Value"]
v_smoothed = v.ewm_mean(alpha=0.03, adjust=False)
v_min = v.rolling_min(w, min_periods=0).ewm_mean(alpha=0.03, adjust=False)
v_max = v.rolling_max(w, min_periods=0).ewm_mean(alpha=0.03, adjust=False)

print("steps", steps[-1])

fig, ax = plt.subplots()
fig.set_size_inches(7, 4)

ax.plot(steps, v_smoothed)
ax.fill_between(steps, v_min, v_max, alpha=0.5)

ax.set_xlabel("Environment steps")
ax.set_ylabel("Score")
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
plt.savefig(f"carchess/plot.svg", bbox_inches='tight')
