import sys
import polars
from matplotlib import pyplot as plt

file = sys.argv[1]

raw = polars.read_csv(file)
steps = raw["Step"]

w = 30

v = raw["Value"]
v_smoothed = v.ewm_mean(alpha=0.03, adjust=False)
mean = v.rolling_mean(w, min_periods=0)
std = v.rolling_std(w, min_periods=0)
v_min = v_smoothed - std * 1.96
v_max = v_smoothed + std * 1.96

s = 500

print("steps", steps[-1])

print("start", steps[-s])
print("prev", steps[-s-1])

def mean_std(data):
    print(f"Mean: {data.mean():.2f}, Stddev: {data.std():.2f}")
mean_std(v[:-s])

fig, ax = plt.subplots()
fig.set_size_inches(7, 4)

ax.plot(steps, mean)
ax.fill_between(steps, v_min, v_max, alpha=0.5)

ax.set_xlabel("Environment steps")
ax.set_ylabel("Score")
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
plt.savefig(f"carchess/plot.svg", bbox_inches='tight')
