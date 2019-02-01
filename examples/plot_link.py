import argparse
import time

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from matplotlib import rc

from gsoftmax.modules import Gspace2d

rc('font', **{'family': 'sans-serif'})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsmath}')

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--device', type=str, help='device')

args = parser.parse_args()


def make_random_alpha(batch_size, h, w):
    alpha = torch.empty((batch_size, h, w)).uniform_(0, 1.)
    supp = torch.empty((batch_size, h, w)).uniform_(0, 1.) > 0.9
    alpha[~supp] = 0
    alpha /= alpha.view(batch_size, -1).sum(dim=1)[:, None, None]
    return alpha


def make_unidim_alpha(batch_size, h, w):
    alpha = torch.zeros((batch_size, h, w))
    alpha[:, h // 5, w // 5: (4 * w) // 5] = 1
    alpha[:, (4 * h) // 5, w // 5: (4 * w) // 5] = 1
    alpha[:, w // 5: (4 * w) // 5, w // 5] = 1
    alpha[:, w // 5: (4 * w) // 5, (4 * w) // 5] = 1
    alpha /= alpha.view(batch_size, -1).sum(dim=1)[:, None, None]
    return alpha


device = args.device
h, w = 32, 32
batch_size = 1
alpha = make_unidim_alpha(batch_size, h, w)
alpha += make_random_alpha(batch_size, h, w) / 5
alpha /= alpha.view(batch_size, -1).sum(dim=1)[:, None, None]
alpha = alpha[:, None]

alpha = alpha.to(device)

gspace = Gspace2d(h, w, sigma=3, max_iter=1000, tol=0,
                  method='lbfgs', logspace=True, verbose=20, )
metric_space = gspace.to(device)
alpha = alpha.to(device)

alpha = alpha.view(batch_size, -1)
v, f = metric_space.entropy_and_potential(alpha)
for i in range(1):
    t0 = time.perf_counter()
    v, proj = gspace.lse_and_softmax(f)
    elapsed = time.perf_counter() - t0
    print(f'Time: {elapsed:.2f}')

sm = F.softmax(f.view(batch_size, -1), dim=1).view(batch_size, 1, h, w)

alpha = alpha.view(batch_size, 1, h, w)
proj = proj.view(batch_size, 1, h, w)
f = f.view(batch_size, 1, h, w)

alpha = alpha.cpu().numpy()
proj = proj.cpu().numpy()
f = f.cpu().numpy()

fig, axes = plt.subplots(1, 4, figsize=(5, 1.2))
fig.subplots_adjust(left=0.0, right=1, top=1, bottom=0.2, wspace=0.02)
axes = axes.ravel()
axes[0].imshow(alpha[0, 0])

axes[0].annotate(r'Distribution $\alpha$', xy=(0.5, -0.1), ha='center',
                 va='top',
                 xycoords='axes fraction'
                 )
axes[1].imshow(-f[0, 0])
axes[1].annotate('Distance field \n' + r'$f = \nabla \Omega(\alpha)$',
                 va='top', xy=(0.5, -0.1), ha='center',
                 xycoords='axes fraction'
                 )
axes[2].imshow(proj[0, 0])
axes[2].annotate('Link function \n' + r'$\nabla \Omega^\star(f)$',
                 xy=(0.5, -0.1), ha='center',
                 va='top',
                 xycoords='axes fraction'
                 )
axes[3].imshow(sm[0, 0])
axes[3].annotate('Naive \n' + r'$\text{softmax}(f)$', xy=(0.5, -0.1),
                 ha='center',
                 va='top',
                 xycoords='axes fraction')
for ax in axes:
    ax.axis('off')
plt.savefig('link.pdf')
