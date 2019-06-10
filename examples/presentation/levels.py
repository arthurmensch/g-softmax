import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib import colors
from matplotlib import rc
from matplotlib.colors import ListedColormap
from torch.nn import Parameter

from gsoftmax.modules import Gspace1d

rc('font', **{'family': 'sans-serif'})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsmath}')
rc('figure.constrained_layout', use=True)
rc('xtick', labelsize=6)
rc('ytick', labelsize=6)

resx = 51
resy = 51


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


fig, axes = plt.subplots(1, 3, figsize=(4, 1.5),
                         gridspec_kw=dict(width_ratios=[1, 1, .05], wspace=0.02))
axes = np.ravel(axes)
cmap = ListedColormap(sns.diverging_palette(255, 133,
                                            l=60, n=50, center="dark"))

for j, c in enumerate([1, 10]):
    C = torch.tensor([[0., c], [c, 0.]])

    gspace = Gspace1d(C, verbose=True)

    xs = np.log(np.linspace(1e-3, 2, resx))  # exp(f)
    ys = np.log(np.linspace(1e-3, 2, resy))  # exp(g)

    f = Parameter(torch.tensor([[x, y] for x in xs for y in ys]))
    distance, alpha = gspace.lse_and_softmax(f)

    alpha = Parameter(alpha.detach().clone())
    v, f_proj = gspace.entropy_and_potential(alpha)

    f_proj += distance[:, None]

    alpha = alpha.detach().numpy().reshape((resx, resy, 2))
    v = v.detach().numpy().reshape((resx, resy))
    distance = distance.detach().numpy()
    distance = distance.reshape((resx, resy))
    distance = np.exp(distance) - 1

    f_proj = f_proj.detach().numpy()
    f = f.detach().numpy()

    f = np.exp(f)
    f_proj = np.exp(f_proj)
    xs = np.exp(xs)
    ys = np.exp(ys)

    ### Plot
    m = axes[j].contourf(xs, ys, distance, 50,
                         norm=MidpointNormalize(midpoint=0.),
                         cmap=cmap, vmin=-1., vmax=3
                         )

    sub = 10 if j == 0 else 20
    f = f.reshape(resx, resy, 2)[::sub, ::sub, :].reshape(-1, 2)
    f_proj = f_proj.reshape(resx, resy, 2)[::sub, ::sub, :].reshape(-1, 2)

    axes[j].scatter(f[:, 0], f[:, 1], color='orange', marker='+', s=2)
    axes[j].scatter(f_proj[:, 0], f_proj[:, 1], color='red', marker='+',
                    s=2)
    plotted = False
    for i in range(len(f)):
        if abs(f_proj[i, 0] - f[i, 0]) > 0.01 or abs(
                f_proj[i, 1] - f[i, 1]) > 0.01:
            plotted += 1
            if plotted == 8:
                axes[j].annotate('$f$', xy=f[i], xytext=(-3, -10),
                                 xycoords='data',
                                 textcoords='offset points',
                                 color='orange')
                axes[j].annotate('$f^E$', xy=f_proj[i], xytext=(-3, -10),
                                 xycoords='data',
                                 textcoords='offset points',
                                 color='red')
            axes[j].arrow(f[i, 0], f[i, 1], f_proj[i, 0] - f[i, 0],
                          f_proj[i, 1] - f[i, 1], color='orange',
                          shape='full', lw=0.1,
                          length_includes_head=True, head_width=0.05,
                          zorder=10)
    if j == 1:
        axes[j].annotate('$f = f^E$', xy=(0.5, 0.1), xytext=(0, +10),
                         fontsize=7,
                         ha='center',
                         xycoords='axes fraction',
                         textcoords='offset points',
                         color='red')
    axes[j].set_ylabel(r'$e^{f_2}$')

    axes[j].annotate(f'$\gamma = {c}$', xy=(.6, .85), fontsize=6,
                     xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc='0.9',
                               ec="black", ))

    axes[j].set_xlim([xs[0], xs[-1]])
    axes[j].set_ylim([ys[0], ys[-1]])

    axes[j].set_xlabel(r'$e^{f_1}$')

# alpha = np.linspace(0, 1, 50)
# beta = np.array(alpha, copy=True)
# beta[0] = .5
# ent = alpha * np.log(beta)
# ent = ent + ent[::-1]
# axes[1].plot(alpha, - ent, color='C0', label=r'$\gamma = \infty$')
# xs = np.linspace(-3, 3, 50)
# axes[0].plot(xs, 1 / (1 + np.exp(-xs)), color='C0')

cbar = fig.colorbar(m, cax=axes[2], extend='both', orientation='vertical')
cbar.set_ticks([-1, 0, 1, 2, 3])
cbar.ax.tick_params(labelsize=5)
axes[2].annotate(r'$\Omega^\star(f)$', (-1.2, -.3), xycoords='axes fraction')
#
# for i, c in enumerate([.1, 2]):
#     C = torch.tensor([[0., c], [c, 0.]])
#     gspace = Gspace1d(C, verbose=True)
#
#     f = Parameter(torch.tensor([[x, 0] for x in xs]))
#     distance, alpha = gspace.lse_and_softmax(f)
#
#     alpha = alpha.numpy()
#     axes[0].plot(xs, alpha[:, 0], color=f'C{i + 1}')
#
#     alpha_0 = torch.linspace(0, 1, 50)
#     alpha = torch.cat([alpha_0[:, None], 1 - alpha_0[:, None]], dim=1)
#     v = gspace.entropy(alpha)
#     v = v.numpy()
#     axes[1].plot(alpha_0.numpy(), -v, color=f'C{i + 1}', label=f'$\gamma = {c}$')
#
# axes[0].set_xlabel(r'$f_1 - f_2$')
# axes[0].set_ylabel(r'${\nabla \Omega^\star(f)}_1$')
#
# axes[1].set_xlabel(r'$\alpha$')
# axes[1].set_ylabel(r'$-\Omega([\alpha, 1-\alpha])$')
#
# axes[1].legend(frameon=True, fontsize=6, loc='lower right')
#
# axes[0].annotate(r'$C =$', xy=(0.7, .45), xycoords='axes fraction',
#                  ha='center', )
# axes[0].annotate(r'$\begin{pmatrix}0 & \gamma \\ \gamma & 0\end{pmatrix}$',
#                  xy=(0.7, .1), ha='center',
#                  xycoords='axes fraction')
#
# sns.despine(fig, ax=axes[0])
# sns.despine(fig, ax=axes[1])

plt.savefig('levels.pdf')
