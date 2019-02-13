import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import Parameter

from gsoftmax.continuous import MeasureDistance
from gsoftmax.sampling import draw_samples, display_samples

n_points = 100
lr = .1
t1 = 10
from_grid = True
flow = 'lifted'


if not from_grid:
    x, a = draw_samples("data/density_a.png", 190, random_state=0)
    a *= 2
else:
    g1 = np.linspace(0, 1, 20)
    g2 = np.linspace(0, 1, 20)
    grid = np.meshgrid(g1, g2)
    x = np.concatenate((grid[0][:, :, None], grid[1][:, :, None]), axis=2)
    x = x.reshape((-1, 2))
    a = np.ones(len(x)) / len(x)

x = torch.from_numpy(x).float()
a = torch.from_numpy(a).float()
a = a[None, :]
x = x[None, :]

y, b = draw_samples("data/density_b.png", 200, random_state=0)
y = torch.from_numpy(y).float()
b = torch.from_numpy(b).float()
b = b[None, :]
y = y[None, :]


sinkhorn_divergence = MeasureDistance(loss='sinkhorn',
                                      coupled=True,
                                      terms='symmetric',
                                      distance_type=2,
                                      kernel='energy_squared',
                                      max_iter=100,
                                      rho=1,
                                      sigma=1, graph_surgery='',
                                      verbose=False,
                                      epsilon=1e-3)

# Parameters for the gradient descent
n_steps = int(np.ceil(t1 / lr))
times = np.linspace(0, n_steps * lr, n_steps + 1)
display_its = np.floor(np.linspace(0, n_steps, 5).astype('int'))

x = Parameter(x)
a = Parameter(a)

plt.figure(figsize=(12, 8))
k = 1
for i, t in enumerate(times):  # Euler scheme ===============
    # Compute cost and gradient
    loss = sinkhorn_divergence(x, a, y, b)
    if x.grad is not None:
        x.grad.zero_()
    loss.backward()
    if flow == 'wasserstein':
        g = x.grad * lr / a[:, :, None]
    else:
        g = x.grad * lr / a.shape[1]
    g[~torch.isfinite(g)] = 0
    info = f't = {t:.3f}, loss {loss.item():.4f}'
    print(info)
    if i in display_its:  # display
        ax = plt.subplot(2, 3, k)
        k = k + 1
        display_samples(ax, y[0], b[0], [(.55, .55, .95)])
        display_samples(ax, x.detach()[0], a.detach()[0],
                        [(.95, .55, .55)], g[0],
                        width=.25 / len(x[0]), scale=5)

        ax.set_title(info)
        ax.axis("equal")
        ax.axis([0, 1, 0, 1])
        plt.xticks([], [])
        plt.yticks([], [])
        ax.set_aspect('equal', adjustable='box')

    x.data -= g
    if flow == 'lifted':
        g = a.grad * lr / a.shape[1]
        g[~torch.isfinite(g)] = 0
        a.data -= g
        a.data = a.data.clamp_(min=0.)
plt.show()
