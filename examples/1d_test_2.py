import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib import transforms

from gsoftmax.lagrangian.sinkhorn import sym_potential, pointwise_eval_potential, quadratic_grad, pairwise_distance, \
    potentials
from gsoftmax.lbfgs_ls import LBFGS
from torch.nn import Parameter

sigma = 0.2

np.random.seed(0)
torch.manual_seed(0)


def density1(x):
    a = 3 * np.exp(-.5 * ((x - 1) / sigma) ** 2)
    a += np.exp(-.5 * ((x - 2) / sigma) ** 2)
    a += np.exp(- .5 * (x / sigma) ** 2)
    a /= np.sum(a, axis=1)
    return a


def density2(x):
    a = np.exp(-.5 * ((x + 1.5)) ** 2)
    a /= np.sum(a, axis=1)
    return a

def density3(x):
    a = np.exp(-.5 * ((x + 2) / sigma) ** 2)
    a += np.exp(-.5 * ((x - 2) / sigma) ** 2)
    a += np.exp(- .5 * (x / sigma) ** 2)
    a /= np.sum(a, axis=1)
    return a

y_ = np.linspace(-3, 3, 200)
y = y_[None, :]
b1 = density3(y)
b2 = density3(y)
y = np.concatenate([y[..., None], np.zeros_like(y)[..., None]], axis=2)

b1 = torch.from_numpy(b1)
b2 = torch.from_numpy(b2)
y = torch.from_numpy(y)

sigma = 1
epsilon = 1e-2
g1, g2 = potentials(y, b1, y, b2, 2, 2, sigma, epsilon, None, 4000, 0)
print(g1, g2)
K = pairwise_distance(y, y, 2, 2, sigma)

g1 = g1[0]
g2 = g2[0]
b1 = b1[0]
b2 = b2[0]
K = K[0]

plan = torch.exp((g1[None, :] + g2[:, None] + K) / epsilon + torch.log(b1)[None, :] + torch.log(b2)[:, None])

print(- (plan * K).sum())
b1 = b1.numpy()
b2 = b2.numpy()
plan = plan.numpy()

fig, axes = plt.subplots(2, 2, figsize=(4, 4),
                         gridspec_kw=dict(width_ratios=[1, 4],
                                          height_ratios=[1, 4],
                                          wspace=0, hspace=0),
                         constrained_layout=False)
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
axes[0, 1].plot(y_, b1)
base = axes[1, 0].transData
rot = transforms.Affine2D().rotate_deg(90)
axes[1, 0].plot(y_, b2, transform=rot + base)

axes[1, 1].contourf(plan, cmap=plt.get_cmap('Reds'), levels=10)
# axes[1, 1].contourf(plan, cmap=plt.get_cmap('Reds'), levels=10)

for ax in axes.ravel():
    ax.axis('off')

plt.savefig('self.svg')
plt.show()