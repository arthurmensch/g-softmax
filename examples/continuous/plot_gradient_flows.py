import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import Parameter
from torch.optim import Adam, SGD

from gsoftmax.continuous import MeasureDistance, mass_scaler
from gsoftmax.sampling import draw_samples, display_samples

n_points = 100
lr = 1e-5
t1 = 1
from_grid = True
flow = 'lifted'

if not from_grid:
    x, a = draw_samples("data/density_a.png", 190, random_state=0)
    a[:] = 0
else:
    g1 = np.linspace(0, 1, 20)
    g2 = np.linspace(0, 1, 20)
    grid = np.meshgrid(g1, g2)
    x = np.concatenate((grid[0][:, :, None], grid[1][:, :, None]), axis=2)
    x = x.reshape((-1, 2))
    a = np.ones(len(x)) / len(x)
    a[:] = 1e-7

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
                                      rho=None,
                                      sigma=1, graph_surgery='',
                                      verbose=False,
                                      epsilon=1e-3)

# Parameters for the gradient descent
n_steps = 5
times = np.linspace(0, 1, n_steps + 1)
display_its = np.floor(np.linspace(0, n_steps, 5).astype('int'))

x = Parameter(x)
a = Parameter(a)

optimizer = Adam([dict(params=[x],
                       lr=1e-1),
                  dict(params=[a],
                       lr=1e-3)])

plt.figure(figsize=(12, 8))
k = 1
for i, t in enumerate(times):  # Euler scheme ===============
    # Compute cost and gradient
    optimizer.zero_grad()
    x = x
    with torch.autograd.detect_anomaly():
        loss = sinkhorn_divergence(x, a, y, b)
        loss.backward()
    torch.nn.utils.clip_grad_norm_([x, a], 10)
    info = f't = {t:.3f}, loss {loss.item():.4f}'
    if i in display_its:  # display
        ax = plt.subplot(2, 3, k)
        k = k + 1
        display_samples(ax, y[0], b[0], [(.55, .55, .95)])
        display_samples(ax, x.detach()[0], a.detach()[0],
                        [(.95, .55, .55)], x.grad[0],
                        width=.25 / len(x[0]), scale=5)

        ax.set_title(info)
        ax.axis("equal")
        ax.axis([0, 1, 0, 1])
        plt.xticks([], [])
        plt.yticks([], [])
        ax.set_aspect('equal', adjustable='box')
    print(info)
    optimizer.step()
    a.data = a.data.clamp_(min=1e-7)

plt.show()
