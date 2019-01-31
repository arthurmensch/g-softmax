import torch
from .impl import _BaseGSpaceImpl, gSoftmaxFunc, gLSEFunc, \
    gEntropyFunc, gPotentialFunc, _GSpace2dImpl, _GSpace1dImpl
from torch.nn import Parameter


class _BaseGSpace(torch.nn.Module):
    def __init__(self, epsilon=1., tol=1e-9,
                 max_iter=1000,
                 verbose=False, method='lbfgs'):
        super().__init__()
        self._impl = _BaseGSpaceImpl(epsilon=epsilon, tol=tol,
                                     max_iter=max_iter, verbose=verbose,
                                     method=method
                                     )

    def to(self, device):
        self._impl.to(device)
        return self

    def softmax(self, f):
        return gSoftmaxFunc.apply(f, self._impl)

    def lse(self, f):
        return gLSEFunc.apply(f, self._impl)

    def lse_and_softmax(self, f):
        f = Parameter(f)
        with torch.enable_grad():
            lse = gLSEFunc.apply(f, self._impl)
        softmax, = torch.autograd.grad(lse.sum(), (f,))
        return lse, softmax

    def entropy(self, alpha):
        return gEntropyFunc.apply(alpha, self._impl)

    def potential(self, alpha):
        return gPotentialFunc.apply(alpha, self._impl)

    def entropy_and_potential(self, alpha):
        alpha = Parameter(alpha)
        with torch.enable_grad():
            lse = gEntropyFunc.apply(alpha, self._impl)
        potentials, = torch.autograd.grad(lse.sum(), (alpha,))
        return lse, potentials

    def hausdorff(self, input, target, reduction='mean'):
        batch_size, h, w = input.shape
        true_entropy = self.entropy(input)
        pred_entropy, potential = self.entropy_and_potential(target)
        bregman = true_entropy - pred_entropy - torch.sum(
            (potential * (input - target)).view(batch_size, -1), dim=1)
        if reduction == 'mean':
            return bregman.mean()
        elif reduction == 'sum':
            return bregman.sum()
        else:
            raise ValueError


class GSpace2d(_BaseGSpace):
    def __init__(self, h, w, n_channels=1,
                 sigma=1., tol=1e-9,
                 max_iter=1000,
                 verbose=False, method='lbfgs'):
        super().__init__()
        self._impl = _GSpace2dImpl(h, w, n_channels,
                                   sigma, tol=tol,
                                   max_iter=max_iter, verbose=verbose,
                                   method=method
                                   )


class GSpace1d(_BaseGSpace):
    def __init__(self, C, tol=1e-9,
                 max_iter=1000,
                 verbose=False, method='lbfgs'):
        super().__init__()
        self._impl = _GSpace1dImpl(C, tol=tol, max_iter=max_iter,
                                   verbose=verbose, method=method)
