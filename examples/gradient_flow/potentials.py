from sampling import draw_samples
import numpy as np
import torch

from gsoftmax.continuous import MeasureDistance

x, a = draw_samples("data/density_b.png", 200, random_state=0)
x = torch.from_numpy(x).float()
a = torch.from_numpy(a).float()
a = torch.log(a)
a = a[None, :]
x = x[None, :]

distance = MeasureDistance(kernel='laplacian_l2',
                           loss='sinkhorn_decoupled',
                           max_iter=100,
                           sigma=1,
                           epsilon=1e-4)
y1 = np.linspace(0, 1, 100)
y2 = np.linspace(0, 1, 100)
y = np.meshgrid(y1, y2)
y = np.concatenate((y[0][:, :, None], y[1][:, :, None]), axis=2)
y = y.reshape((-1, 2))
y = torch.from_numpy(y).float()[None, :]

f, fe = distance.potential(x, a, y)

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2)
axes[0].scatter(x[0, :, 0], x[0, :, 1])
axes[1].contour(y1, y2, fe[0].reshape(len(y1), len(y2)), 50
                )
axes[0].set_xlim([0, 1])
axes[0].set_ylim([0, 1])
axes[1].set_xlim([0, 1])
axes[1].set_ylim([0, 1])

plt.show()
