import torch
import torch.nn as nn
import torch.nn.functional as F


def duality_gap(f, fn):
    diff = f - fn
    return (torch.max(diff, dim=1)[0] - torch.min(diff, dim=1)[
        0]).mean().item()


def opp(dir):
    if dir in ['xx', 'yy']:
        return dir
    elif dir == 'xy':
        return 'yx'
    else:
        return 'xy'


def bmm(x, y):
    return torch.einsum('blk,bk->bl', x, y)


class MeasureDistance(nn.Module):
    def __init__(self, max_iter=10, sigma=1, tol=1e-6,
                 kernel='gaussian', epsilon=1,
                 loss='sinkhorn',
                 verbose=False):
        """

        :param max_iter:
        :param sigma:
        :param tol:
        :param kernel: str in ['l1', 'l2', 'l22', 'gaussian', 'laplacian']
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
        self.kernel = kernel
        self.verbose = verbose
        assert loss in ['sinkhorn', 'sinkhorn_decoupled',
                        'sinkhorn_decoupled_asym',
                        'mmd', 'mmd_decoupled', 'mmd_decoupled_asym']

        assert kernel in ['l1', 'l2', 'l22', 'gaussian', 'laplacian',
                          'laplacian_l2']

        if 'mmd' in loss:
            self.epsilon = 1
            # assert kernel != 'l22'

        self.loss = loss

    def make_kernel(self, x: torch.tensor, y: torch.tensor):
        """

        :param x: shape(batch_size, l, d)
        :param y: shape(batch_size, k, d)
        :return: kernel: shape(batch_size, l, k)
            Negative kernel matrix
        """
        if self.kernel in ['l22', 'l2', 'gaussian', 'laplacian_l2']:
            outer = torch.einsum('bld,bkd->blk', x, y)
            norm_x = torch.sum(x ** 2, dim=2)
            norm_y = torch.sum(y ** 2, dim=2)
            distance = ((norm_x[:, :, None] + norm_y[:, None, :]
                         - 2 * outer)
                        / 2 / self.sigma ** 2)
            distance.clamp_(min=0.)
            if self.kernel == 'l22':
                return - distance
            elif self.kernel == 'l2':
                return - torch.sqrt(distance + 1e-8)
            elif self.kernel == 'gaussian':
                return torch.exp(-distance)
            elif self.kernel == 'laplacian_l2':
                return torch.exp(- torch.sqrt(distance + 1e-8))
        elif self.kernel in ['l1', 'laplacian']:
            diff = x[:, :, None, :] - y[:, None, :, :]
            distance = (torch.sum(torch.abs(diff), dim=3)
                        / self.sigma)
            if self.kernel == 'l1':
                return - distance
            else:
                return torch.exp(-distance)
        else:
            raise ValueError

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
        kernels, potentials = {}, {}

        kernels['xx'] = self.make_kernel(x, x)

        if y is not None or self.loss in ['mmd', 'sinkhorn']:
            kernels['yx'] = self.make_kernel(y, x)
        if self.loss in ['mmd', 'sinkhorn']:
            kernels['xy'] = kernels['yx'].transpose(1, 2)
            kernels['yy'] = self.make_kernel(y, y)

        if 'mmd' in self.loss:
            a = a.exp()
            b = b.exp()
            potentials['xx'] = - bmm(kernels['xx'], a) / 2
            if y is not None or 'loss' == 'mmd':
                potentials['yx'] = - bmm(kernels['xy'], x) / 2
            if 'decoupled' in self.loss:
                if y is not None:  # Extrapolate
                    return potentials['xx'], potentials['yx']
                return potentials['xx']
            potentials['xy'] = - bmm(kernels['xy'], b) / 2
            potentials['yy'] = - bmm(kernels['yy'], b) / 2
            return (potentials['xy'] - potentials['xx'],
                    potentials['yx'] - potentials['yy'])

        kernels['xx'] = kernels['xx'] / self.epsilon + a[:, None, :].detach()
        potentials = dict(xx=torch.zeros_like(a))
        runnings = dict(xx=True)

        if y is not None or not 'decoupled' in self.loss:
            kernels['yx'] = kernels['yx'] / self.epsilon + a[:, None,
                                                           :].detach()

        if not 'decoupled' in self.loss:
            potentials = dict(xx=potentials['xx'], yx=torch.zeros_like(b),
                              xy=torch.zeros_like(a), yy=torch.zeros_like(b))
            runnings = dict(xx=True, yx=True, xy=True, yy=True)
            kernels['xy'] = kernels['xy'] / self.epsilon + b[:, None,
                                                           :].detach()
            kernels['yy'] = kernels['yy'] / self.epsilon + b[:, None,
                                                           :].detach()

        grad_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(False)
        n_iter = 0
        while any(runnings.values()) and n_iter < self.max_iter:
            for dir, running in runnings.items():
                if running:
                    new_potential = - torch.logsumexp(
                        kernels[dir] + potentials[opp(dir)][:, None, :],
                        dim=2)  # xy -> yx, yx -> xy, xx -> xx, yy -> yy
                    gap = duality_gap(new_potential, potentials[dir])
                    # print(f'dir {dir} gap {gap}')
                    if gap < self.tol:
                        runnings[dir] = False
                    potentials[dir] *= .5
                    potentials[dir] += .5 * new_potential
            if self.loss == 'sinkhorn':
                runnings['xy'] = runnings['yx'] = runnings['yx'] or runnings[
                    'xy']
            n_iter += 1
        torch.set_grad_enabled(grad_enabled)
        if 'decoupled' in self.loss and y is not None:
            potentials['xy'] = potentials['xx']
            # so that potentials[opp('yx')] = potentials['xx']
        potentials_extra = {}
        for dir in kernels:  # Extrapolation step
            potentials_extra[dir] = - torch.logsumexp(
                kernels[dir] + potentials[opp(dir)].detach()[:, None, :],
                dim=2)
        if 'decoupled' in self.loss:
            if y is not None:
                return potentials_extra['xx'], potentials_extra['yx']
            else:
                return potentials_extra['xx']
        else:
            return (potentials_extra['xy'] - potentials_extra['xx'],
                    potentials_extra['yx'] - potentials_extra['yy'])

    def forward(self, x: torch.tensor, a: torch.tensor,
                y: torch.tensor, b: torch.tensor):
        if 'decoupled' in self.loss:
            f, fe = self.potential(x, a, y)
            g, ge = self.potential(y, b, x)
            res = torch.sum((ge - f) * a.exp(), dim=1)

            if 'asym' not in self.loss:
                sym = torch.sum((ge - f) * a.exp(), dim=1)
                res = (res + sym) / 2
        else:
            f, g = self.potential(x, a, y, b)
            res = torch.sum(f * a.exp(), dim=1) + torch.sum(g * b.exp(), dim=1)
        return res * self.epsilon


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
