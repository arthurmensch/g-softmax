from gsoftmax.sampling import draw_samples
import numpy as np
import torch

from gsoftmax.continuous import MeasureDistance

x, a = draw_samples("data/density_b.png", 200, random_state=0)
x = torch.from_numpy(x).float()
x[:, 0] -= .2
a = torch.from_numpy(a).float()
a = torch.log(a)
a = a[None, :]
x = x[None, :]

y, b = draw_samples("data/density_a.png", 200 + 10, random_state=1)
y = torch.from_numpy(y).float()
b = torch.from_numpy(b).float()
b = torch.log(b)
b = b[None, :]
y = y[None, :]

distance = MeasureDistance(kernel='energy', loss='sinkhorn',
                           coupled=True, verbose=True,
                           distance_type=2, max_iter=1000, tol=1e-8,
                           sigma=1, epsilon=1e-4)
g1 = np.linspace(0, 1, 100)
g2 = np.linspace(0, 1, 100)
grid = np.meshgrid(g1, g2)
grid = np.concatenate((grid[0][:, :, None], grid[1][:, :, None]), axis=2)
grid = grid.reshape((-1, 2))
grid = torch.from_numpy(grid).float()[None, :]

if not distance.coupled:
    f, fe = distance.potential(x, a, grid)
    g, ge = distance.potential(y, b, grid)
else:
    f, g = distance.potential(x, a, y, b)
    fe = distance.extrapolate(potential=f, target_pos=grid, pos=x, weight=a)
    ge = distance.extrapolate(potential=g, target_pos=grid, pos=y, weight=b)
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 2)
axes[0, 0].scatter(x[0, :, 0], x[0, :, 1])
axes[0, 1].contour(g1, g2, fe[0].reshape(len(g1), len(g2)), 50)
axes[1, 0].scatter(y[0, :, 0], y[0, :, 1])
axes[1, 1].contour(g1, g2, ge[0].reshape(len(g1), len(g2)), 50)
for ax in axes.ravel():
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

plt.show()
