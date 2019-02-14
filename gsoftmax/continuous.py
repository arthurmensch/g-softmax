import math
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList


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
                 terms: str = 'symmetric', epsilon: float = 1., rho=None,
                 tol: float = 1e-6, max_iter: int = 10,
                 kernel: str = 'energy_squared', distance_type: int = 2,
                 graph_surgery: str = 'loop',
                 sigma: float = 1, verbose: bool = False,
                 reduction='mean'):
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
        self.rho = rho

        self.verbose = verbose
        assert loss in ['sinkhorn', 'mmd']
        self.loss = loss
        self.coupled = coupled
        assert terms in ['left', 'right', 'symmetric', ['left', 'right'],
                         ['right', 'left']]
        if terms == 'symmetric':
            terms = ['left', 'right']
        elif terms in ['left', 'right']:
            terms = [terms]
        self.terms = terms
        self.graph_surgery = graph_surgery

        assert distance_type in [1, 2]
        self.distance_type = distance_type
        if self.distance_type == 2:
            assert kernel in ['energy', 'energy_squared',
                              'gaussian', 'laplacian']
        else:
            assert kernel in ['energy', 'laplacian']
        self.kernel = kernel

        assert reduction in ['none', 'mean', 'sum']
        self.reduction = reduction

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
            divider = 2 if self.kernel == 'gaussian' else 1
            distance = ((norm_x[:, :, None] + norm_y[:, None, :] - 2 * outer) /
                        (divider * self.sigma ** 2))
            distance.clamp_(min=0.)

            if self.kernel in ['energy', 'laplacian']:
                distance = torch.sqrt(distance + 1e-8)
            distance = - distance
            if self.kernel in ['energy', 'energy_squared']:
                return distance
            elif self.kernel in ['laplacian', 'gaussian']:
                multiplier = 2 if self.kernel == 'laplacian' else math.sqrt(2 * math.pi)
                return torch.exp(distance) / multiplier / self.sigma
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
                  y: torch.tensor = None, b: torch.tensor = None,):
        """

        :param x: shape(batch_size, l, d)
        :param a: shape(batch_size, l)
        :param y: shape(batch_size, k, d)
        :param b: shape(batch_size, k)
        :return:
        """

        def detach_pos(z):
            return z.detach() if 'pos' in self.graph_surgery else z

        def detach_weight(z):
            return z.detach() if 'weight' in self.graph_surgery else z

        kernels, potentials, new_potentials = {}, {}, {}
        pos, weights, gaps = {}, {}, {}

        weights['x'], pos['x'] = detach_weight(a), x

        running = ['xx'] if not self.coupled else ['xx', 'yy', 'xy', 'yx']

        if b is not None:
            weights['y'] = detach_weight(b)
        if y is not None:
            pos['y'] = y

        kernels['xx'] = self.make_kernel(pos['x'], detach_pos(pos['x']))
        if self.coupled or y is not None:
            kernels['yx'] = self.make_kernel(pos['y'], detach_pos(pos['x']))
        if self.coupled:
            kernels['yy'] = self.make_kernel(pos['y'], detach_pos(pos['y']))
            kernels['xy'] = self.make_kernel(pos['x'], detach_pos(pos['y']))

        if self.loss == 'mmd':
            for dir in kernels:
                potentials[dir] = self.extrapolate(kernel=kernels[dir],
                                                   weight=weights[dir[1]])
            if not self.coupled:
                if y is not None:
                    return potentials['xx'], potentials['yx']
                return potentials['xx']
            return (potentials['xx'], potentials['yx'],
                    potentials['yy'], potentials['xy'])

        weights = {dir: torch.log(weight) for dir, weight in weights.items()}

        for dir in running:
            potentials[dir] = torch.zeros_like(weights[dir[0]])

        n_iter = 0
        with torch.no_grad() if 'loop' in self.graph_surgery else nullcontext():
            while running and n_iter < self.max_iter:
                for dir in running:
                    new_potentials[dir] = self.extrapolate(
                        potential=potentials[dir[::-1]],
                        weight=weights[dir[1]], epsilon=self.epsilon,
                        kernel=kernels[dir])
                for dir in running:
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
        for dir in kernels:  # Extrapolation step
            new_potentials[dir] = self.extrapolate(
                potential=potentials[dir[::-1]],
                weight=weights[dir[1]],
                epsilon=self.epsilon,
                kernel=kernels[dir])
        new_potentials = {dir: self.potential_transform(potential)
                          for dir, potential in new_potentials.items()}
        if not self.coupled:
            if y is not None:
                return new_potentials['xx'], new_potentials['yx']
            return new_potentials['xx']
        return (new_potentials['xx'], new_potentials['yx'],
                new_potentials['yy'], new_potentials['xy'])

    def extrapolate(self, potential: torch.tensor = None,
                    weight: torch.tensor = None, kernel: torch.tensor = None,
                    pos: torch.tensor = None, target_pos: torch.tensor = None,
                    epsilon: float = None):
        if kernel is None:
            kernel = self.make_kernel(target_pos, pos)
        if self.loss == 'mmd':
            return - bmm(kernel, weight) / 2
        if epsilon is None:
            epsilon = self.epsilon
        scale = 1 if self.rho is None else 1 / (1 + self.epsilon / self.rho)
        return - epsilon * scale * torch.logsumexp(
            (kernel + potential[:, None, :]) / epsilon
            + weight[:, None, :], dim=2)

    def potential_transform(self, f):
        if self.rho is None or self.loss == 'mmd':
            return f
        else:
            # scale * rho
            return - (self.rho + self.epsilon / 2) * (- f / self.rho).exp()

    def forward(self, x: torch.tensor, a: torch.tensor,
                y: torch.tensor, b: torch.tensor):

        if not self.coupled:
            f, fe = self.potential(x, a, y)
            g, ge = self.potential(y, b, x)
        else:
            f, fe, g, ge = self.potential(x, a, y, b)
        res = 0
        if 'right' in self.terms:
            diff = ge - f
            diff[a == 0] = 0
            res += torch.sum(diff * a, dim=1)
        if 'left' in self.terms:
            diff = fe - g
            diff[b == 0] = 0
            res += torch.sum(diff * b, dim=1)

        if self.reduction == 'mean':
            res = res.mean()
        elif self.reduction == 'sum':
            res = res.sum()
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
    def __init__(self, beads=10, dimension=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2),
            # nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 5, padding=2),
            # nn.BatchNorm2d(16),
            nn.Conv2d(16, 64, 2, stride=2),
            # nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 256, 2, stride=2),
            # nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 4, stride=4),
            # nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        self.weight_fc = nn.Sequential(
            nn.Linear(4096, 2048),
            # nn.BatchNorm1d(2048),
            nn.ReLU(True),
            nn.Linear(2048, 2048),
            # nn.BatchNorm1d(2048),
            nn.ReLU(True),
            nn.Linear(2048, 2048),
            # nn.BatchNorm1d(2048),
            nn.ReLU(True),
            nn.Linear(2048, 2048),
            # nn.BatchNorm1d(2048),
            nn.ReLU(True),
            nn.Linear(2048, beads)
        )

        self.pos_fc = nn.Sequential(
            nn.Linear(4096, 2048),
            # nn.BatchNorm1d(2048),
            nn.ReLU(True),
            nn.Linear(2048, 2048),
            # nn.BatchNorm1d(2048),
            nn.ReLU(True),
            nn.Linear(2048, 2048),
            # nn.BatchNorm1d(2048),
            nn.ReLU(True),
            nn.Linear(2048, 2048),
            # nn.BatchNorm1d(2048),
            nn.ReLU(True),
            nn.Linear(2048, beads * dimension),

        )

        self.dimension = dimension
        self.beads = beads

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(-1, 4096)
        position = torch.sigmoid(self.pos_fc(x).reshape(-1, self.beads, self.dimension) * 10)
        weights = self.weight_fc(x)
        weights = F.sigmoid(weights)
        return position, weights
