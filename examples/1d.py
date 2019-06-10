import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from gsoftmax.lagrangian.sinkhorn import sym_potential, pointwise_eval_potential, quadratic_grad
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


def plot(b, g, a, y, x, i):
    fig, axes = plt.subplots(1, 4, figsize=(8, 2), constrained_layout=True)
    values = quadratic_grad(y, x, a, y, b, g, 2, 2, sigma, 1, None)
    b = b.numpy()[0]
    g = g.numpy()[0]
    a = a.numpy()[0]
    y = y.numpy()[0, :, 0]
    x = x.numpy()[0, :, 0]
    values = values.numpy()[0]
    axes[0].plot(y, g, linewidth=2)
    axes[0].annotate('Score function\n' + r'$g_\theta(x)$', xy=(.5, 0.7), ha='center',
                     xycoords='axes fraction')
    axes[1].plot(y, b, color='red', linewidth=2)
    axes[1].annotate('Target distribution\n' + r'$\alpha = \nabla \Omega^*(g_\theta(x))$',
                     xy=(.5, 0.7), ha='center', xycoords='axes fraction')
    axes[2].scatter(x, a, marker='+')
    axes[2].annotate('Frank-Wolfe\nestimation ' + r'$\alpha_t$',
                     xy=(.5, 0.7), ha='center', xycoords='axes fraction')
    axes[3].plot(y, values, color='orange', linewidth=2)
    axes[3].annotate('Linear minimization\noracle ' + rf'$t = {i}$',
                     xy=(.5, 0.7), ha='center', xycoords='axes fraction')
    axes[2].set_ylim([0, 0.5])
    axes[3].set_ylim([0, 5])
    axes[0].set_ylim([0, 5])
    axes[1].set_ylim([0, 0.05])
    axes[1].set_yticks([])
    axes[2].set_yticks([])
    for j in range(4):
        axes[j].set_xlim([-3, 3])

    sns.despine(fig, axes)
    plt.savefig(f'1d_{i}.pdf')


y = np.linspace(-3, 3, 200)[None, :]
b = density(y)
y = np.concatenate([y[..., None], np.zeros_like(y)[..., None]], axis=2)

b = torch.from_numpy(b)
y = torch.from_numpy(y)

g = sym_potential(y, b, 2, 2, sigma, 1, None, 1000, 1e-5)
ge = pointwise_eval_potential(y, y, b, g, 2, 2, sigma, 1, None, )

x_l = np.linspace(-3, 3, 10)[:, None]
x_l = np.concatenate([x_l, np.zeros_like(x_l)], axis=1)
x_l = torch.from_numpy(x_l)
x_l = [this_x[None, :] for this_x in x_l]

a_l = (np.ones(len(x_l)) / len(x_l)).tolist()

x = torch.cat(x_l, dim=0)
x = x[None, :, :]
a = torch.tensor(a_l, dtype=torch.float64)
a = a[None, :]

values = quadratic_grad(y, x, a, y, b, g, 2, 2, sigma, 1, None)

for i in range(100):
    x = torch.cat(x_l, dim=0)
    x = x[None, :, :]
    a = torch.tensor(a_l, dtype=torch.float64)
    a = a[None, :]
    pos = (torch.rand(1) - 0.5) * 5
    z = Parameter(torch.tensor([[pos.item(), 0]], dtype=torch.float64))

    def value_grad():
        res = quadratic_grad(z[None, :], x, a, y, b, g, 2, 2, sigma, 1, None)
        res = res[0, 0]
        res.backward()
        return res


    optimizer = LBFGS([z], lr=1, line_search_fn='strong_Wolfe')
    optimizer.step(closure=value_grad)
    x_l.append(z.data)
    a_l = [this_a * (1 - 2 / (i + 3)) for this_a in a_l]
    a_l.append(2 / (i + 3))

    if i % 10 == 0:
        print(i)
        plot(b, g, a, y, x, i)
