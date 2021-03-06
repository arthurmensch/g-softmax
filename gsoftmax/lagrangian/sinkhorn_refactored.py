from contextlib import nullcontext

import torch
from torch import nn as nn


def duality_gap(potential, new_potential):
    """
    Duality gap in Sinkhorn.

    :param potential
    :param new_potential
    :return:
    """
    with torch.no_grad():
        diff = potential - new_potential
        res = torch.max(diff, dim=1)[0] - torch.min(diff, dim=1)[0]
        res[torch.isnan(res)] = 0
        return res.mean().item()


def bmm(x: torch.tensor, y):
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
                 graph_surgery: str = 'loop+weight',
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
            distance = ((norm_x[:, :, None] + norm_y[:, None, :]
                         - 2 * outer))
            distance.clamp_(min=0.)

            if self.kernel == 'energy':
                return - torch.sqrt(distance + 1e-8)
            elif self.kernel == 'energy_squared':
                return - distance
            elif self.kernel == 'laplacian':
                return torch.exp(- torch.sqrt(distance + 1e-8) / self.sigma)
            elif self.kernel == 'gaussian':
                return torch.exp(- distance / 2 / self.sigma ** 2)
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
                  y: torch.tensor = None, b: torch.tensor = None, ):
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

        scale = 1 if self.rho is None else 1 + self.epsilon / self.rho
        n_iter = 0
        with torch.no_grad() if 'loop' in self.graph_surgery else nullcontext():
            while running and n_iter < self.max_iter:
                for dir in running:
                    new_potentials[dir] = self.extrapolate(
                        potential=potentials[dir[::-1]],
                        weight=weights[dir[1]], epsilon=self.epsilon,
                        kernel=kernels[dir]) / scale
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
                kernel=kernels[dir]) / scale
        scale = 1 if self.rho is None else 1 + self.epsilon / self.rho / 2
        new_potentials = {dir: phi_transform(potential, self.rho) * scale
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
        potential = masker(potential, weight == -float('inf'), -float('inf'))
        sum = potential / epsilon + weight
        operand = kernel / epsilon + sum[:, None, :]
        lse = - epsilon * safe_lse(operand)
        return lse

    def forward(self, x: torch.tensor, a: torch.tensor,
                y: torch.tensor, b: torch.tensor):
        if not self.coupled:
            f, fe = self.potential(x, a, y)
            g, ge = self.potential(y, b, x)
        else:
            f, fe, g, ge = self.potential(x, a, y, b)
            # fe is the usual g, ge is the usual f, f and g are symmetric pots
        res = 0
        if 'right' in self.terms:
            ge = masker(ge, a == 0, 0)
            f = masker(f, a == 0, 0)
            res += torch.sum(ge * a, dim=1) - torch.sum(f * a, dim=1)
        if 'left' in self.terms:
            fe = masker(fe, b == 0, 0)
            g = masker(g, b == 0, 0)
            res += torch.sum(fe * b, dim=1) - torch.sum(g * b, dim=1)

        if self.reduction == 'mean':
            res = res.mean()
        elif self.reduction == 'sum':
            res = res.sum()
        return res


def safe_lse(operand):
    return SafeLSE.apply(operand)


class SafeLSE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, operand):
        ctx.save_for_backward(operand)
        return torch.logsumexp(operand, dim=2)

    @staticmethod
    def backward(ctx, grad_output):
        operand,  = ctx.saved_tensors
        mask = torch.all(torch.isinf(operand), dim=2) ^ 1
        s = torch.ones_like(operand) / operand.shape[2]
        s[mask] = torch.softmax(operand[mask], dim=1)
        return s * grad_output[:, :, None]


def masker(x, mask, target):
    return Masker.apply(x, mask, target)


class Masker(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask, target):
        x[mask] = target
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def mass_scaler(x, a):
    return MassScaler.apply(x, a)


class MassScaler(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a):
        ctx.save_for_backward(a)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        grad_output = grad_output * 1
        return grad_output, None


def phi_transform(f, rho):
    if rho is None:
        return f
    else:
        f = f.clone()
        mask = torch.isfinite(f)
        f[mask] = - rho * ((- f[mask] / rho).exp() - 1)
        return f