import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib import transforms

from gsoftmax.lagrangian.sinkhorn import sym_potential, pointwise_eval_potential, quadratic_grad, pairwise_distance
from gsoftmax.lbfgs_ls import LBFGS
from torch.nn import Parameter

sigma = 0.2

np.random.seed(0)
torch.manual_seed(0)


def density(x):
    a = np.exp(-.5 * ((x + 2) / sigma) ** 2)
    a += np.exp(-.5 * ((x - 2) / sigma) ** 2)
    a += np.exp(- .5 * (x / sigma) ** 2)
    a /= np.sum(a, axis=1)
    return a


y_ = np.linspace(-3, 3, 200)
y = y_[None, :]
b = density(y)
y = np.concatenate([y[..., None], np.zeros_like(y)[..., None]], axis=2)

b = torch.from_numpy(b)
y = torch.from_numpy(y)

sigma = 1e-3
epsilon = 10
g = sym_potential(y, b, 2, 2, sigma, epsilon, None, 1000, 0)
K = pairwise_distance(y, y, 2, 2, sigma)

g = g[0]
b = b[0]
K = K[0]

plan = torch.exp((g[None, :] + g[:, None] - K) / epsilon) * (b[None, :] * b[:, None])

print((plan * K).sum())
b = b.numpy()
plan = plan.numpy()

fig, axes = plt.subplots(2, 2, figsize=(4, 4),
                         gridspec_kw=dict(width_ratios=[1, 4],
                                          height_ratios=[1, 4],
                                          wspace=0, hspace=0),
                         constrained_layout=False)
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
axes[0, 1].plot(y_, b)
base = axes[1, 0].transData
rot = transforms.Affine2D().rotate_deg(90)
axes[1, 0].plot(y_, b, transform=rot + base)

axes[1, 1].matshow(plan, cmap=plt.get_cmap('Reds'))

for ax in axes.ravel():
    ax.axis('off')

plt.savefig('test.svg')
plt.show()