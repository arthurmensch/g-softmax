import matplotlib.pyplot as plt
import torch
from torch.nn import Parameter

from gsoftmax.continuous import MeasureDistance
from gsoftmax.sampling import draw_samples, display_samples

n_points = 200

x, a = draw_samples("data/density_a.png", n_points)
x = torch.from_numpy(x).float()
a = torch.from_numpy(a).float()
a = torch.log(a)
a = a[None, :]
x = x[None, :]

y, b = draw_samples("data/density_b.png", n_points + 10)
y = torch.from_numpy(y).float()
b = torch.from_numpy(b).float()
b = torch.log(b)
b = b[None, :]
y = y[None, :]

plt.figure(figsize=(7, 7))
plt.scatter([10], [10])  # shameless hack to prevent change of axis

display_samples(plt.gca(), y[0], [(.55, .55, .95)])
display_samples(plt.gca(), x[0], [(.95, .55, .55)])

plt.axis("equal")
plt.axis([0, 1, 0, 1])
plt.gca().set_aspect('equal', adjustable='box')


def gradient_flow(x, a, y, b, cost, lr=.05):
    """
    Flows along the gradient of the cost function, using a simple Euler scheme.
    
    Parameters
    ----------
        α_i : (N,1) torch tensor
            weights of the source measure
        x : (N,2) torch tensor
            samples of the source measure
        β_j : (M,1) torch tensor
            weights of the target measure
        y_j : (M,2) torch tensor
            samples of the target measure
        cost : (a,x,beta_j,y_j) -> torch float number,
            real-valued function
        lr : float, default = .05
            learning rate, i.e. time step
    """

    # Parameters for the gradient descent
    Nsteps = int(5 / lr) + 1
    display_its = [int(t / lr) for t in [0, .25, .50, 1., 2., 5.]]

    # Make sure that we won't modify the input measures
    # We're going to perform gradient descent on Cost(Alpha, Beta)
    # wrt. the positions x of the diracs masses that make up Alpha:
    x = Parameter(x)

    plt.figure(figsize=(12, 8))
    k = 1
    for i in range(Nsteps):  # Euler scheme ===============
        # Compute cost and gradient
        loss = cost(x, a, y, b)
        if x.grad is not None:
            x.grad.zero_()
        loss.backward()
        g = x.grad
        print(g.min())
        print(f'{loss.item():.4f}')

        if i in display_its:  # display
            ax = plt.subplot(2, 3, k)
            k = k + 1
            ax.scatter([10], [10])  # shameless hack

            display_samples(ax, y[0], [(.55, .55, .95)])
            display_samples(ax, x[0], [(.95, .55, .55)],
                            g[0] / a.exp()[0][:, None],
                            width=.25 / len(x[0]), scale=5)

            ax.set_title("t = {:1.2f}".format(lr * i))
            ax.axis("equal")
            ax.axis([0, 1, 0, 1])
            plt.xticks([], [])
            plt.yticks([], [])
            ax.set_aspect('equal', adjustable='box')

        # in-place modification of the tensor's values
        x.data -= lr * (g / a.exp()[:, :, None])
    print("Done.")


sinkhorn_divergence = MeasureDistance(kernel='energy_squared',
                                      loss='sinkhorn',
                                      coupled=False,
                                      distance_type=2,
                                      max_iter=100,
                                      sigma=1,
                                      epsilon=1e-3)

gradient_flow(x, a, y, b, sinkhorn_divergence, lr=.1)

plt.show()
