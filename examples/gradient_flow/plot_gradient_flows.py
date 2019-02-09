import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import Parameter

from gsoftmax.continuous import MeasureDistance
from gsoftmax.sampling import draw_samples, display_samples

n_points = 200
lr = .1
time = 5

x, a = draw_samples("data/density_a.png", n_points, random_state=0)
x = torch.from_numpy(x).float()
a = torch.from_numpy(a).float()
a = torch.log(a)
a = a[None, :]
x = x[None, :]

y, b = draw_samples("data/density_b.png", n_points + 10, random_state=0)
y = torch.from_numpy(y).float()
b = torch.from_numpy(b).float()
b = torch.log(b)
b = b[None, :]
y = y[None, :]


sinkhorn_divergence = MeasureDistance(loss='sinkhorn',
                                      coupled=True,
                                      symmetric=True,
                                      # target_position='left',
                                      distance_type=2,
                                      kernel='energy_squared',
                                      max_iter=100,
                                      sigma=1,
                                      verbose=True,
                                      epsilon=1e-4)

# Parameters for the gradient descent
n_steps = int(time / lr)

display_its = np.floor(np.linspace(0, n_steps - 1, 5)).astype('int')

x = Parameter(x)

plt.figure(figsize=(12, 8))
k = 1
for i in range(n_steps):  # Euler scheme ===============
    # Compute cost and gradient
    loss = sinkhorn_divergence(x, a, y, b)
    if x.grad is not None:
        x.grad.zero_()
    loss.backward()
    g = x.grad / a.exp()[:, :, None] * lr
    print(f'{loss.item():.4f}')

    if i in display_its:  # display
        ax = plt.subplot(2, 3, k)
        k = k + 1
        display_samples(ax, y[0], [(.55, .55, .95)])
        display_samples(ax, x[0], [(.95, .55, .55)], g[0],
                        width=.25 / len(x[0]), scale=5)

        ax.set_title(f"t = {i / (n_steps - 1):.3f}")
        ax.axis("equal")
        ax.axis([0, 1, 0, 1])
        plt.xticks([], [])
        plt.yticks([], [])
        ax.set_aspect('equal', adjustable='box')

    # in-place modification of the tensor's values
    x.data -= g

plt.show()
