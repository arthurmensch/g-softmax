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


def _opp(dir):
    if dir in ['xx', 'yy']:
        return dir
    elif dir == 'xy':
        return 'yx'
    else:
        return 'xy'


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
                 kernel: str = 'energy', distance_type: int = 2,
                 target_position='left',
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
        epsilon = 1 if self.loss == 'mmd' else self.epsilon

        if self.distance_type == 2:
            outer = torch.einsum('bld,bkd->blk', x, y)
            norm_x = torch.sum(x ** 2, dim=2)
            norm_y = torch.sum(y ** 2, dim=2)
            divider = 1 if self.kernel in ['energy', 'laplacian'] else 2
            power = 2 if self.kernel in ['energy', 'laplacian'] else 1
            distance = ((norm_x[:, :, None] + norm_y[:, None, :] - 2 * outer) /
                        (divider * self.sigma ** 2 * epsilon ** power))
            distance.clamp_(min=0.)

            if self.kernel in ['energy', 'laplacian']:
                distance = torch.sqrt(distance + 1e-8)
                if self.kernel == 'energy':
                    return - distance
                elif self.kernel == 'laplacian':
                    return torch.exp(-distance)
            elif self.kernel == 'energy_squared':
                return - distance
            elif self.kernel == 'gaussian':
                return torch.exp(-distance)
            raise ValueError(f'Wrong kernel argument for'
                             f' distance_type==2, got `{self.kernel}`')
        elif self.distance_type == 1:
            diff = x[:, :, None, :] - y[:, None, :, :]
            distance = ((torch.sum(torch.abs(diff), dim=3))
                        / (self.sigma * epsilon))
            if self.kernel == 'energy':
                return - distance
            elif self.kernel == 'laplacian':
                return torch.exp(-distance)
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
        if not self.coupled and y is not None:
            return self.potential(x, a), self.potential(y, b)

        kernels, potentials = {}, {}

        kernels['xx'] = self.make_kernel(x, x)

        if self.coupled:
            kernels['yx'] = self.make_kernel(y, x)
            kernels['xy'] = kernels['yx'].transpose(1, 2)
            kernels['yy'] = self.make_kernel(y, y)

        if self.loss == 'mmd':
            a = a.exp()
            potentials['xx'] = - bmm(kernels['xx'], a) / 2
            if not self.coupled:
                return potentials['xx']
            b = b.exp()
            potentials['yx'] = - bmm(kernels['yx'], a) / 2
            potentials['xy'] = - bmm(kernels['xy'], b) / 2
            potentials['yy'] = - bmm(kernels['yy'], b) / 2
            return (potentials['xy'] - potentials['xx'],
                    potentials['yx'] - potentials['yy'])

        kernels['xx'] = kernels['xx'] + a[:, None, :]
        potentials = dict(xx=torch.zeros_like(a))
        runnings = dict(xx=True)

        if self.coupled:
            kernels['yx'] = kernels['yx'] + a[:, None, :]
            kernels['xy'] = kernels['xy'] + b[:, None, :]
            kernels['yy'] = kernels['yy'] + b[:, None, :]
            potentials = dict(xx=potentials['xx'], yx=torch.zeros_like(b),
                              xy=torch.zeros_like(a), yy=torch.zeros_like(b))
            runnings = dict(xx=True, yx=True, xy=True, yy=True)

        # with torch.no_grad():
        n_iter = 0
        while any(runnings.values()) and n_iter < self.max_iter:
            for dir, running in runnings.items():
                if running:
                    new_potential = self.extrapolate(potentials[_opp(dir)],
                                                     kernel=kernels[dir])
                    gap = duality_gap(new_potential, potentials[dir])
                    print(dir, gap)
                    if gap < self.tol:
                        runnings[dir] = False
                    potentials[dir] *= .5
                    potentials[dir] += .5 * new_potential
            if self.coupled:
                runnings['yx'] = runnings['yx'] or runnings['xy']
                runnings['xy'] = runnings['yx']
            n_iter += 1
        potentials_extra = {}
        for dir in kernels:  # Extrapolation step
            potentials_extra[dir] = self.extrapolate(potentials[_opp(dir)],
                                                     kernel=kernels[dir])
        if self.coupled:
            return (potentials_extra['xy'] - potentials_extra['xx'],
                    potentials_extra['yx'] - potentials_extra['yy'])
        else:
            return potentials_extra['xx']

    def extrapolate(self, potential: torch.tensor,
                    x: torch.tensor = None, a: torch.tensor = None,
                    y: torch.tensor = None, kernel: torch.tensor = None):
        if kernel is None:
            kernel = self.make_kernel(y, x) / self.epsilon + a[:, None, :]
        return - torch.logsumexp(kernel + potential[:, None, :], dim=2)

    def forward(self, x: torch.tensor, a: torch.tensor,
                y: torch.tensor, b: torch.tensor):
        epsilon = 1 if self.loss == 'mmd' else self.epsilon
        if 'decoupled' in self.loss:
            if self.target_position == 'left' and not self.symmetric:
                x, a, y, b = y, b, x, a
            f = self.potential(x, a)
            fe = self.extrapolate(f, x=x, a=a, y=y)
            g = self.potential(y, b)
            left = torch.sum((fe - g) * b.exp(), dim=1)
            if not self.symmetric:
                return left * epsilon
            ge = self.extrapolate(g, x=y, a=b, y=x)
            right = torch.sum((ge - f) * a.exp(), dim=1)
            return (left + right) * (epsilon / 2)
        f, g = self.potential(x, a, y, b)
        res = torch.sum(f * a.exp(), dim=1) + torch.sum(g * b.exp(), dim=1)
        return res * epsilon


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
