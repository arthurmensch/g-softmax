import numpy as np
import torch
from scipy.optimize import fmin_l_bfgs_b
from torch.optim import Optimizer

eps = np.finfo('double').eps


class LBFGS(Optimizer):
    """Wrap L-BFGS algorithm, using scipy routines.

    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).

    .. warning::
        Right now CPU only

    .. note::
        This is a very memory intensive optimizer (it requires additional
        ``param_bytes * (history_size + 1)`` bytes). If it doesn't fit in memory
        try reducing the history size, or use a different algorithm.

    Arguments:
        max_iter (int): maximal number of iterations per optimization step
            (default: 20)
        max_eval (int): maximal number of function evaluations per optimization
            step (default: max_iter * 1.25).
        tolerance_grad (float): termination tolerance on first order optimality
            (default: 1e-5).
        tolerance_change (float): termination tolerance on function
            value/parameter changes (default: 1e-9).
        history_size (int): update history size (default: 100).
    """

    def __init__(self, params, max_iter=20, max_eval=None,
                 tolerance_grad=1e-5, tolerance_change=1e-9, history_size=10,
                 device='cpu', dtype=torch.double,
                 verbose=False, bounds=(None, None),
                 ):
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        defaults = dict(max_iter=max_iter, max_eval=max_eval,
                        tolerance_grad=tolerance_grad, tolerance_change=tolerance_change,
                        device=device, dtype=dtype,
                        history_size=history_size, bounds=bounds)
        super(LBFGS, self).__init__(params, defaults)

        self._n_iter = 0
        self._last_loss = None

        self.verbose = verbose

        self._prepare_swaps()


    def _prepare_swaps(self):
        numel = 0
        for group in self.param_groups:
            for p in group['params']:
                numel += p.numel()
            device = group['device']
        self._grad = torch.zeros(numel, dtype=torch.double)
        self._params = torch.zeros(numel, dtype=torch.double)
        if device.type == 'cuda':
            self._grad = self._grad.pin_memory()
            self._params = self._params.pin_memory()

    def _gather_flat_grad(self):
        offset = 0
        for group in self.param_groups:
            for p in group['params']:
                grad = p.grad.cpu().double()
                numel = grad.numel()
                if grad.is_sparse:
                    grad = grad.data.to_dense().view(-1)
                else:
                    grad = grad.data.view(-1)
                self._grad[offset:offset + numel] = grad
                offset += numel
        return self._grad

    def _gather_flat_params(self):
        offset = 0
        for group in self.param_groups:
            for p in group['params']:
                p = p.data.cpu().double()
                numel = p.numel()
                if p.data.is_sparse:
                    p = p.to_dense().view(-1)
                else:
                    p = p.view(-1)
                self._params[offset:offset + numel] = p
                offset += numel
        return self._params

    def _gather_flat_bounds(self):
        bounds = []
        for group in self.param_groups:
            n = 0
            for p in group['params']:
                n += p.numel()
            bounds += [group['bounds']] * n
        return bounds

    def _distribute_flat_params(self, params):
        offset = 0
        for group in self.param_groups:
            device = group['device']
            dtype = group['dtype']
            params = params.type(dtype).to(device)
            for p in group['params']:
                numel = p.numel()
                # view as to avoid deprecated pointwise semantics
                p.data = params[offset:offset + numel].view_as(p)
                offset += numel

    def reset_params(self, params):
        self.param_groups[0]['params'] = params

    def step(self, closure):
        """Performs a single optimization step.

        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        group = self.param_groups[0]
        max_iter = group['max_iter']
        max_eval = group['max_eval']
        tolerance_grad = group['tolerance_grad']
        tolerance_change = group['tolerance_change']
        history_size = group['history_size']

        def wrapped_closure(flat_params):
            """closure must call zero_grad() and backward()"""
            flat_params = torch.from_numpy(flat_params)
            self._distribute_flat_params(flat_params)
            loss = closure()
            self._last_loss = loss
            loss = loss.item()
            flat_grad = self._gather_flat_grad().numpy()
            return loss, flat_grad

        def callback(flat_params):
            self._n_iter += 1
            if self.verbose == 1:
                print('Iter %i Loss %.5f' % (self._n_iter, self._last_loss.item()))

        bounds = self._gather_flat_bounds()
        initial_params = self._gather_flat_params()

        if max_iter > 0:
            fmin_l_bfgs_b(wrapped_closure, x0=initial_params, maxiter=max_iter,
                          maxfun=max_eval, maxls=20,
                          bounds=bounds,
                          factr=tolerance_change / eps, pgtol=tolerance_grad,
                          epsilon=0,
                          disp=self.verbose > 1,
                          m=history_size,
                          callback=callback)
