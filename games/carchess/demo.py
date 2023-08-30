import matplotlib.pyplot as plt  # type: ignore [import]
import matplotlib.animation as animation  # type: ignore [import]

from .map import Map

m = Map("map1")
m.reset()

fig, ax = plt.subplots()

m.plot_grid(ax)

artists = []

for step in range(20):
    artists.append(m.plot(ax))
    m.simulation_step()

ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=1000)

plt.show()
