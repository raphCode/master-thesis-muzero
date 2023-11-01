import sys
import polars
import glob
import os
import numpy as np
import pathlib
from collections import defaultdict
from matplotlib import pyplot as plt

final_samples, data_name, folder = sys.argv[1:]
final_samples = int(final_samples)

steps = None
plot_data = defaultdict(list)
score_data = defaultdict(list)
thres_data = defaultdict(list)
counts = defaultdict(int)
labels = dict()

thres = 0

files = glob.glob(f"{folder}/**/{data_name}.csv", recursive=True)
for file in files:
    print("Reading", file)
    p = pathlib.Path(file).parts
    variant = p[1]
    raw = polars.read_csv(file)
    v = raw["Value"]
    v_smoothed=v.ewm_mean(alpha=0.03, adjust=False)
    s = np.array(raw["Step"])
    if steps is not None:
        assert np.allclose(s, steps)
    steps = s
    plot_data[variant].append(v_smoothed)
    with open(os.path.join(*p[:2] , "label"), "r") as f:
        labels[variant] = f.readline().strip()

    counts[variant] += 1

    samples = np.array(v)[-final_samples:]
    score_data[variant] = np.concatenate([score_data[variant], samples])

    first_idx = np.argmax(np.array(v_smoothed) >thres)
    thres_data[variant].append(int(s[first_idx]))

print("samples:", list(score_data.values())[0].size)
print("first step:",steps[-final_samples])

fig, ax = plt.subplots()
fig.set_size_inches(7, 4)

for variant, runs in plot_data.items():
    label=labels[variant]
    aaa = np.stack(runs)
    mean = aaa.mean(axis=0)
    stddev = aaa.std(axis=0)
    ci = 1.96 * stddev
    ci = 1.645 * stddev
    ax.plot(steps, mean, label=label)
    ax.fill_between(steps, mean - ci, mean + ci, alpha=0.5)

    def mean_std(tag, data):
        print(tag, f"Mean: {data.mean():.2f}, Stddev: {data.std():.2f}")
    print(label, counts[variant], "runs")
    mean_std("final score:", score_data[variant])
    mean_std(f"thres {thres}:", np.array(thres_data[variant]))


ax.legend(loc="lower right")
ax.set_ylim([-1.1, 1.1])
ax.set_xlabel("Environment steps")
ax.set_ylabel("Score")
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
plt.savefig(f"{folder}/plot.svg", bbox_inches='tight')
