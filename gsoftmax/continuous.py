import torch
import torch.nn as nn
import torch.nn.functional as F


def duality_gap(potential, new_potential):
    """
    Duality gap in Sinkhorn.

    :param potential
    :param new_potential
    :return:
    """
    diff = potential - new_potential
    return (torch.max(diff, dim=1)[0] - torch.min(diff, dim=1)[
        0]).mean().item()


def bmm(x, y):
    """
    Batched matrix multiplication

    :param x:
    :param y:
    :return:
    """
    return torch.einsum('blk,bk->bl', x, y)


class MeasureDistance(nn.Module):
    def __init__(self, loss: str = 'sinkhorn', coupled: bool = True,
                 symmetric: bool = True, epsilon: float = 1.,
                 tol: float = 1e-6, max_iter: int = 10,
                 kernel: str = 'energy_squared', distance_type: int = 2,
                 target_position='right',
                 sigma: float = 1, verbose: bool = False):
        """

        :param max_iter:
        :param sigma:
        :param tol:
        :param kernel: str in ['energy', 'gaussian', 'laplacian']
        :param loss: str in ['sinkhorn', 'sinkhorn_decoupled',
                             'sinkhorn_decoupled_asym',
                             'mmd', 'mmd_decoupled', 'mmd_decoupled_asym']
        :param verbose:
        """
        super().__init__()

        self.max_iter = max_iter
        self.tol = tol
        self.sigma = sigma
        self.epsilon = epsilon

        self.verbose = verbose
        assert loss in ['sinkhorn', 'mmd']
        self.loss = loss
        self.coupled = coupled
        self.symmetric = symmetric
        self.target_position = target_position

        assert distance_type in [1, 2]
        self.distance_type = distance_type
        if self.distance_type == 2:
            assert kernel in ['energy', 'energy_squared',
                              'gaussian', 'laplacian']
        else:
            assert kernel in ['energy', 'laplacian']
        self.kernel = kernel

    def make_kernel(self, x: torch.tensor, y: torch.tensor):
        """

        :param x: shape(batch_size, l, d)
        :param y: shape(batch_size, k, d)
        :return: kernel: shape(batch_size, l, k)
        """
        if self.distance_type == 2:
            outer = torch.einsum('bld,bkd->blk', x, y)
            norm_x = torch.sum(x ** 2, dim=2)
            norm_y = torch.sum(y ** 2, dim=2)
            divider = 1 if self.kernel in ['energy', 'laplacian'] else 2
            distance = ((norm_x[:, :, None] + norm_y[:, None, :] - 2 * outer) /
                        (divider * self.sigma ** 2))
            distance.clamp_(min=0.)

            if self.kernel in ['energy', 'laplacian']:
                distance = torch.sqrt(distance + 1e-8)
            distance = - distance
            if self.kernel in ['energy', 'energy_squared']:
                return distance
            elif self.kernel in ['laplacian', 'gaussian']:
                return torch.exp(distance)
            raise ValueError(f'Wrong kernel argument for'
                             f' distance_type==2, got `{self.kernel}`')
        elif self.distance_type == 1:
            diff = x[:, :, None, :] - y[:, None, :, :]
            distance = ((torch.sum(torch.abs(diff), dim=3)) / self.sigma)
            if self.kernel == 'energy':
                return - distance
            elif self.kernel == 'laplacian':
                return torch.exp(- distance)
            else:
                raise ValueError(f'Wrong kernel argument for'
                                 f' distance_type==1, got `{self.kernel}`')
        else:
            raise ValueError(f'Wrong distance_type argument, '
                             f'got `{self.distance_type}`')

    def potential(self, x: torch.tensor, a: torch.tensor,
                  y: torch.tensor = None,
                  b: torch.tensor = None,
                  ):
        """

        :param x: shape(batch_size, l, d)
        :param a: shape(batch_size, l)
        :param y: shape(batch_size, k, d)
        :param b: shape(batch_size, k)
        :return:
        """
        kernels, potentials, new_potentials = {}, {}, {}
        pos, weights, gaps = {}, {}, {}

        weights['x'], pos['x'] = a.detach(), x

        running = ['xx'] if not self.coupled else ['xx', 'yy', 'xy', 'yx']

        if b is not None:
            weights['y'] = b.detach()
        if y is not None:
            pos['y'] = y

        kernels['xx'] = self.make_kernel(pos['x'], pos['x'])
        if self.coupled:
            kernels['yy'] = self.make_kernel(pos['y'], pos['y'])
            kernels['xy'] = self.make_kernel(pos['x'], pos['y'])
            kernels['yx'] = kernels['xy'].transpose(2, 1)

        if self.loss == 'mmd':
            for dir in running:
                potentials[dir] = self.extrapolate(kernel=kernels[dir],
                                                   weight=weights[dir[1]])
            if not self.coupled:
                if y is not None:
                    return potentials['xx'], potentials['yx']
                return potentials['xx']
            return (potentials['xy'] - potentials['xx'],
                    potentials['yx'] - potentials['yy'])

        for dir in running:
            potentials[dir] = torch.zeros_like(weights[dir[0]])

        n_iter = 0
        with torch.no_grad():
            while running and n_iter < self.max_iter:
                for dir in running:
                    new_potentials[dir] = self.extrapolate(
                        potential=potentials[dir[::-1]],
                        weight=weights[dir[1]], epsilon=self.epsilon,
                        kernel=kernels[dir])
                for dir in running:
                    # if dir in ['xx', 'yy']:
                    new_potentials[dir] += potentials[dir]
                    new_potentials[dir] /= 2
                    gaps[dir] = duality_gap(new_potentials[dir],
                                            potentials[dir])
                    potentials[dir] = new_potentials[dir]
                running = list(filter(lambda dir:
                                      gaps[dir] > self.tol
                                      or gaps[dir[::-1]] > self.tol,
                                      running))
                if self.verbose:
                    print('Iter', n_iter, 'running', running, 'gaps', gaps)
                n_iter += 1
        if not self.coupled and y is not None:
            potentials['xy'] = potentials['xx']
            kernels['yx'] = self.make_kernel(pos['y'], pos['x'])
        for dir in kernels:  # Extrapolation step
            new_potentials[dir] = self.extrapolate(
                potential=potentials[dir[::-1]],
                weight=weights[dir[1]],
                epsilon=self.epsilon,
                kernel=kernels[dir])
        if not self.coupled:
            if y is not None:
                return new_potentials['xx'], new_potentials['yx']
            return new_potentials['xx']
        return (new_potentials['xy'] - new_potentials['xx'],
                new_potentials['yx'] - new_potentials['yy'])

    def extrapolate(self, potential: torch.tensor = None,
                    weight: torch.tensor = None, kernel: torch.tensor = None,
                    pos: torch.tensor = None, target_pos: torch.tensor = None,
                    epsilon: float = None):
        if kernel is None:
            kernel = self.make_kernel(target_pos, pos)
        if self.loss == 'mmd':
            return - bmm(kernel, weight.exp()) / 2
        if epsilon is None:
            epsilon = self.epsilon

        return - epsilon * torch.logsumexp(
            (kernel + potential[:, None, :]) / epsilon
            + weight[:, None, :], dim=2)

    def forward(self, x: torch.tensor, a: torch.tensor,
                y: torch.tensor, b: torch.tensor):
        if not self.coupled:
            if self.target_position == 'right' and not self.symmetric:
                x, a, y, b = y, b, x, a
            f, fe = self.potential(x, a, y)
            if not self.symmetric:
                return torch.sum(fe * b.exp(), dim=1)
            g, ge = self.potential(y, b, x)
            left = torch.sum((fe - g) * b.exp(), dim=1)
            right = torch.sum((ge - f) * a.exp(), dim=1)
            return (left + right) / 2
        f, g = self.potential(x, a, y, b)
        res = torch.sum(f * a.exp(), dim=1) + torch.sum(g * b.exp(), dim=1)
        return res


class ResLinear(nn.Module):
    def __init__(self, n_features, bias=True):
        super().__init__()
        self.l1 = nn.Linear(n_features, n_features, bias=bias)
        self.l2 = nn.Linear(n_features, n_features, bias=bias)
        self.bn1 = nn.BatchNorm1d(n_features)
        self.bn2 = nn.BatchNorm1d(n_features)

    def forward(self, x):
        return F.relu(self.bn2(self.l2(F.relu(self.bn1(self.l1(x))))) + x)


class DeepLoco(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 5, padding=2),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 256, 2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 4, stride=4),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        self.resnet = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
        )

        self.weight_fc = nn.Linear(2048, 256)
        self.pos_fc = nn.Linear(2048, 256 * 3)

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(-1, 4096)
        x = self.resnet(x)
        pos = F.tanh(self.pos_fc(x).reshape(-1, 256, 3) * 1000)
        weights = F.log_softmax(self.weight_fc(x), dim=1)
        return pos, weights
