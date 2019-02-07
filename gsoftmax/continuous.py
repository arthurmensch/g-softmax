import torch
import torch.nn as nn
import torch.nn.functional as F


class Hausdorff(nn.Module):
    def __init__(self, max_iter=100, sigma=1, tol=1e-6,
                 kernel='gaussian', normalize=False, verbose=False):
        super().__init__()
        self.max_iter = max_iter
        self.tol = tol
        self.sigma = sigma
        self.kernel = kernel
        self.normalize = normalize
        self.verbose = verbose

    def make_kernel(self, x: torch.tensor, y: torch.tensor):
        """

        :param x: shape(batch_size, l, d)
        :param y: shape(batch_size, k, d)
        :return: kernel: shape(batch_size, length, dim)
        """
        if self.kernel == 'gaussian':
            outer = torch.einsum('bld,bkd->blk', x, y)
            norm_x = torch.sum(x ** 2, dim=2)
            norm_y = torch.sum(y ** 2, dim=2)
            return - ((norm_x[:, :, None] + norm_y[:, None, :]
                       - 2 * outer)
                      / 2 / self.sigma ** 2)
        elif self.kernel == 'laplacian':
            diff = torch.einsum('bld,bkd->blkd', x, y)
            return - torch.sum(torch.abs(diff), dim=3) / self.sigma
        else:
            raise ValueError

    def potential(self, pos: torch.tensor, log_weights: torch.tensor, ):
        """

        :param pos: shape (batch_size, length, dim)
        :param log_weights:  shape (batch_size, length)
        :return:
        """
        kernel = self.make_kernel(pos, pos)
        f = torch.zeros_like(log_weights)
        for i in range(self.max_iter):
            g = - torch.logsumexp(kernel + f[:, None, :]
                                  + log_weights[:, None, :], dim=2)
            with torch.no_grad():
                diff = f - g
                gap = (torch.max(diff, dim=1)[0]
                       - torch.min(diff, dim=1)[0]).mean()
                if self.verbose:
                    print(f'[Negentropy] Iter {i}, gap {gap}')
            if gap.mean() < self.tol:
                break
            f = (f + g) / 2
        return f

    def forward(self, pos: torch.tensor, log_weights: torch.tensor,
                target_pos: torch.tensor, target_log_weights: torch.tensor):
        f = self.forward(pos, log_weights)
        kernel = self.make_kernel(target_pos, pos)
        g = - torch.logsumexp(kernel + f[:, None, :]
                              + log_weights[:, None, :], dim=2)
        res = torch.sum(g * torch.exp(target_log_weights), dim=1)
        if self.normalize:
            res -= torch.sum(f * torch.exp(log_weights), dim=1)
        return res


def duality_gap(f, fn):
    diff = f - fn
    return (torch.max(diff, dim=1)[0] - torch.min(diff, dim=1)[0]).mean()


class Sinkhorn(nn.Module):
    def __init__(self, max_iter=100, sigma=1, tol=1e-6,
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

        assert kernel in ['l1', 'l2', 'l22', 'gaussian', 'laplacian']

        if 'mmd' in loss:
            assert epsilon == 1
            assert kernel != 'l22'

        self.loss = loss

    def make_kernel(self, x: torch.tensor, y: torch.tensor):
        """

        :param x: shape(batch_size, l, d)
        :param y: shape(batch_size, k, d)
        :return: kernel: shape(batch_size, l, k)
            Negative kernel matrix
        """
        if self.kernel in ['l22', 'l2', 'gaussian']:
            outer = torch.einsum('bld,bkd->blk', x, y)
            norm_x = torch.sum(x ** 2, dim=2)
            norm_y = torch.sum(y ** 2, dim=2)
            distance = ((norm_x[:, :, None] + norm_y[:, None, :]
                         - 2 * outer)
                        / 2 / self.sigma ** 2 / self.epsilon)
            if self.kernel == 'l22':
                return distance
            elif self.kernel == 'l2':
                return torch.sqrt(distance)
            elif self.kernel == 'gaussian':
                return - torch.exp(-distance)
        elif self.kernel in ['l1', 'laplacian']:
            diff = torch.einsum('bld,bkd->blkd', x, y)
            distance = (torch.sum(torch.abs(diff), dim=3)
                        / self.sigma / self.epsilon)
            if self.kernel == 'l1':
                return distance
            else:
                return - torch.exp(-distance)
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

        kernels['xx'] = self.make_kernels(x, x)

        if y is not None or not 'decoupled' in self.loss:
            kernels['yx'] = self.make_kernels(y, x)
        if not 'decoupled' in self.loss:
            kernels['xy'] = kernels['yx'].transpose(2, 1)
            kernels['yy'] = self.make_kernels(y, y)

        if 'mmd' in self.loss:
            potentials['xx'] = - kernels['xx'][None, :, :].dot(a)
            if y is not None or not 'decoupled' in self.loss:
                potentials['yx'] = - kernels['yx'][None, :, :].dot(a)
            if 'decoupled' in self.loss:
                if y is not None:  # Extrapolate
                    return potentials['xx'], potentials['yx']
                return potentials['xx']
            potentials['xy'] = - kernels['xy'][None, :, :].dot(b)
            potentials['yy'] = - kernels['yy'][None, :, :].dot(b)
            return (potentials['xy'] - potentials['xx'] / 2,
                    potentials['yx'] - potentials['yy'] / 2)

        kernels['xx'] = kernels['xx'] + a[:, None, :]
        potentials = dict(xx=torch.zeros_like(a))
        running = dict(xx=True)

        if not 'decoupled' in self.loss:
            potentials = dict(xx=potentials['xx'], yx=torch.zeros_like(b),
                              xy=torch.zeros_like(a), yy=torch.zeros_like(b))
            running = dict(**running, yx=True, xy=True, yy=True)
            kernels['yx'] = kernels['yx'] + a[:, None, :]
            kernels['xy'] = kernels['xy'] + b[:, None, :]
            kernels['yy'] = kernels['yy'] + b[:, None, :]

        grad_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(False)
        n_iter = 0
        while any(running.values()) and n_iter < self.max_iter:
            for dir, state in running.items():
                if running:
                    new_potential = - torch.logsumexp(
                        kernels[dir] + potentials[reversed(dir)][:, None, :],
                        dim=2)  # xy -> yx, yx -> xy, xx -> xx, yy -> yy
                    gap = duality_gap(new_potential, potentials[dir])
                    if gap < self.tol:
                        running[dir] = False
                    potentials[dir] *= .5
                    potentials[dir] += .5 * new_potential
            if self.loss == 'sinkhorn':
                running['xy'] = running['yx'] = running['yx'] or running['xy']
            n_iter += 1
        torch.set_grad_enabled(grad_enabled)
        for dir in potentials:  # Extrapolation step
            potentials[dir] = - torch.logsumexp(
                kernels[dir] + potentials[reversed(dir)][:, None, :], dim=2)
        if 'decoupled' in self.loss:
            if y is not None:  # Extrapolate
                potentials['yx'] = - torch.logsumexp(
                    kernels['yx'] + potentials['xx'][:, None, :], dim=2)
            return potentials['xx'], potentials['yx']
        else:
            return (potentials['xy'] - potentials['xx'] / 2,
                    potentials['yx'] - potentials['yy'] / 2)

    def forward(self, x: torch.tensor, a: torch.tensor,
                y: torch.tensor, b: torch.tensor):
        if 'decoupled' in self.loss:
            f, g = self.potential(x, a, y)
            res = torch.sum(g * torch.exp(b) - f * torch.exp(a))
            if 'asym' in self.loss:
                return res
            g, f = self.potential(y, b, x)
            res = (res + torch.sum(f * torch.exp(a) - g * torch.exp(b))) / 2
        else:
            f, g = self.potential(x, a, y, b)
            res = torch.sum(f * a, dim=1) + torch.sum(g * b, dim=2)
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
