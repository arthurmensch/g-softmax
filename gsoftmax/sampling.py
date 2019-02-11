import numpy as np
from scipy import misc
from sklearn.utils import check_random_state


def load_image(fname):
    img = misc.imread(fname, flatten=True)  # Grayscale
    img = (img[::-1, :]) / 255.
    # img = np.swapaxes(img, 0,1 )
    return 1 - img


def draw_samples(fname, n, random_state=None):
    random_state = check_random_state(random_state)
    A = load_image(fname)
    xg, yg = np.meshgrid(np.linspace(0, 1, A.shape[0]),
                         np.linspace(0, 1, A.shape[1]))

    grid = np.array(list(zip(xg.ravel(), yg.ravel())))
    dens = A.ravel() / A.sum()
    choice = random_state.choice(np.arange(len(grid)), p=dens, size=n)
    dots = grid[choice]
    dots += (.5 / A.shape[0]) * random_state.standard_normal(dots.shape)

    weights = np.ones(n) / n
    return dots, weights


def display_samples(ax, x, size, color, x_grad=None, scale=None, width=0.0025):
    x_ = x.data.cpu().numpy()
    ax.scatter(x_[:, 0], x_[:, 1], 3000 * size, color)

    if x_grad is not None:
        g_ = -x_grad.data.cpu().numpy()
        if scale is None:
            sc, scu = .05 / len(x_), "dots"
        else:
            sc, scu = scale, "xy"
        ax.quiver(x_[:, 0], x_[:, 1], g_[:, 0], g_[:, 1],
                  scale=sc, scale_units=scu, color="#5CBF3A",
                  zorder=3, width=width)
