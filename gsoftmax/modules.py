import torch
import torch.nn.functional as F
from torch.nn import Parameter

from gsoftmax.lbfgs_ls import LBFGS


def safe_log(y):
    zero_mask = y == 0
    y = y.clone()
    y[zero_mask] = 1
    logy = torch.log(y)
    logy[zero_mask] = - float('inf')
    return logy


class LSEFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f, init, gspace):
        with torch.no_grad():
            value, grad = gspace._lse(f, init)
        ctx.save_for_backward(grad)
        return value

    @staticmethod
    def backward(ctx, grad_output):
        grad, = ctx.saved_tensors
        return grad * grad_output[:, None], None, None


class EntropyFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha, gspace):
        with torch.no_grad():
            value, grad = gspace._entropy(alpha)
        ctx.save_for_backward(grad)
        return value

    @staticmethod
    def backward(ctx, grad_output):
        grad, = ctx.saved_tensors
        return grad * grad_output[:, None], None


class _BaseGspace(torch.nn.Module):
    def __init__(self, tol=1e-9,
                 max_iter=1000,
                 verbose=False,
                 method='lbfgs',
                 ):
        super().__init__()
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

        self.method = method

    def _entropy(self, alpha: torch.tensor):
        with torch.no_grad():
            f = torch.zeros_like(alpha)
            log_alpha = safe_log(alpha)
            for i in range(self.max_iter):
                g = self.c_transform(f, log_alpha)
                diff = f - g
                gap = (torch.max(diff, dim=1)[0]
                       - torch.min(diff, dim=1)[0]).mean()
                if self.verbose:
                    print(f'[Negentropy] Iter {i}, gap {gap}')
                if gap.mean() < self.tol:
                    break
                f = .5 * (f + g)
            f = -f
        return (f * alpha).sum(dim=1), f

    def _lse(self, f: torch.tensor, init=None):
        if self.method == 'lbfgs':
            return self._lse_lbfgs(f, init)
        elif self.method in ['fw', 'fw_ls', 'fw_away', 'fw_pw']:
            return self._lse_fw(f, init)
        else:
            raise ValueError('Wrong method')

    def entropy(self, alpha: torch.tensor):
        return EntropyFunc.apply(alpha, self)

    def lse(self, f, init=None):
        return LSEFunc.apply(f, init, self)

    def lse_and_softmax(self, f, init=None):
        with torch.enable_grad():
            f = Parameter(f)
            lse = self.lse(f, init)
        softmax, = torch.autograd.grad(lse.sum(), (f,))
        return lse, softmax

    def softmax(self, f, init=None):
        return self.lse_and_softmax(f, init=init)[1]

    def entropy_and_potential(self, alpha):
        with torch.enable_grad():
            alpha = Parameter(alpha)
            entropy = self.entropy(alpha)
        potentials, = torch.autograd.grad(entropy.sum(), (alpha,))
        return entropy, potentials

    def potential(self, alpha):
        return self.entropy_and_potential(alpha)[1]

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

    def _log_conv(self, u: torch.tensor):
        raise NotImplementedError

    def _get_C(self, s, v):
        raise NotImplementedError

    def c_transform(self, f: torch.tensor, log_alpha: torch.tensor):
        f = f / 2 + log_alpha
        log_conv = self._log_conv(f)
        return - 2 * log_conv

    def _log_quadratic(self, l, f):
        batch_size, _ = l.shape
        lse = torch.logsumexp(l, dim=1)
        l = l - lse[:, None] + f
        return torch.logsumexp(l + self._log_conv(l), dim=1)

    def _log_quadratic_grad(self, l, f):
        batch_size, _ = l.shape
        lse = torch.logsumexp(l, dim=1)
        l = l - lse[:, None]
        lgrad = self._log_conv(l + f) + f
        return lgrad

    def _lse_lbfgs(self, f: torch.tensor, init=None):
        batch_size, _ = f.shape

        with torch.no_grad():
            fd = (- f / 2).detach()

            if init is not None:
                l = safe_log(init.clone())
            else:
                l = -2 * fd
            l -= torch.logsumexp(l, dim=1)[:, None]

            v_init = - self._log_quadratic(l, fd)
            v_init = v_init.mean().item()

            with torch.enable_grad():
                l = Parameter(l)

                optimizer = LBFGS([l], max_iter=self.max_iter,
                                  lr=1,
                                  tolerance_change=self.tol,
                                  tolerance_grad=0,
                                  line_search_fn='strong_Wolfe')
                self.n_eval = 0

                def closure():
                    optimizer.zero_grad()
                    v = self._log_quadratic(l, fd)
                    v = v.sum()
                    if self.verbose > 10:
                        self.n_eval += 1
                        print(f'Eval {self.n_eval} value {v.item()}')
                    v.backward()
                    return v

                optimizer.step(closure)
            l = l.detach()
            v = - self._log_quadratic(l, fd)
            if self.verbose:
                print(f'Initial quadratic value: {v_init:.3f},'
                      f' maximized value: {v.mean().item():.3f}')
        return v, torch.softmax(l, dim=1)

    def _lse_fw(self, f: torch.tensor, init=None):
        batch_size, _ = f.shape

        with torch.no_grad():
            fd = (- f / 2).detach()

            if init is not None:
                alpha = init.clone()
            else:
                alpha = F.softmax(-2 * fd, dim=1)

            for k in range(self.max_iter):
                l = safe_log(alpha)
                lgrad = self._log_quadratic_grad(l, fd)
                s = torch.min(lgrad, dim=1)[1]
                lsa = lgrad[range(batch_size), s]

                laa = torch.logsumexp(l + lgrad, dim=1)
                m = torch.max(laa, lsa)
                log_gap = m + torch.log(torch.exp(laa - m) - torch.exp(lsa - m))
                gap = torch.exp(log_gap).mean()

                value = - laa

                if gap < self.tol:
                    break

                if self.method in ['fw_away', 'fw_pw']:
                    lgrad_masked = lgrad.clone()
                    lgrad_masked[alpha <= 0] = - float('inf')
                    v = torch.max(lgrad_masked, dim=1)[1]
                    lva = lgrad[range(batch_size), v]
                    m = torch.max(laa, lva)
                    log_gap_away = m + torch.log(
                        torch.exp(lva - m) - torch.exp(laa - m))
                if self.method == 'fw_pw':
                    lss = 2 * fd[range(batch_size), s]
                    lvv = 2 * fd[range(batch_size), v]

                    max_step_size = alpha[range(batch_size), v]

                    lsv = fd[range(batch_size), s] + fd[range(batch_size), v]
                    lsv += self._get_C(s, v)
                    m = torch.cat([l[:, None]
                                   for l in [lsa, lva, lss, lvv, lsv]], dim=1)
                    m = torch.max(m, dim=1)[0]
                    step_size = ((torch.exp(lva - m) - torch.exp(lsa - m))
                                 / (torch.exp(lss - m) + torch.exp(lvv - m)
                                    - 2 * torch.exp(lsv - m)))

                    step_size.clamp_(min=0.)
                    step_size = torch.min(step_size, max_step_size)

                    alpha[range(batch_size), s] += step_size
                    alpha[range(batch_size), v] -= step_size
                else:
                    if self.method == 'fw_away':
                        # Selection between FW step and away step
                        away_mask = log_gap_away >= log_gap

                        these_alpha = alpha[range(batch_size), v]
                        max_step_size = torch.ones_like(these_alpha)

                        max_step_size[away_mask] = these_alpha / (1 - these_alpha)
                        lsa[away_mask] = lva[away_mask]
                        s[away_mask] = v[away_mask]

                    if self.method in ['fw_ls', 'fw_away']:
                        lss = 2 * fd[range(batch_size), s]
                        nomin = lss - lsa
                        denom = laa - lsa
                        m = torch.max(nomin, denom).clamp_(min=0.)
                        step_size = 1 / (
                                1 + (torch.exp(nomin - m) - torch.exp(-m)) /
                                (torch.exp(denom - m) - torch.exp(-m)))
                        if self.method == 'fw_away':
                            step_size[away_mask] *= -1
                        step_size.clamp_(min=0., max=1.)
                        if self.method == 'fw_away':
                            step_size = torch.min(step_size, max_step_size)
                            step_size[away_mask] *= -1
                        alpha *= 1 - step_size[:, None]
                    else:
                        step_size = 2 / (k + 2)
                        alpha *= 1 - step_size
                    alpha[range(batch_size), s] += step_size
                alpha.clamp_(min=0., max=1.)

                mean_step_size = step_size if self.method == "fw" else step_size.mean()

                if self.verbose and k % 10 == 0:
                    info = (f'Iter {k}, gap = {gap:.1f}, '
                            f"value = {value.mean():.5f}, "
                            f's = {s}, '
                            f'step_size = {mean_step_size:.2e}')
                    if self.method in ['fw_pw', 'fw_away']:
                        gap_away = torch.exp(log_gap_away).mean()
                        info += f', gap_away = {gap_away:.1f}'
                    if self.method == 'fw_pw':
                        info += f', v = {v}'
                    print(info)
                if mean_step_size == 0.:
                    break
            value = - self._log_quadratic(safe_log(alpha), fd)
            print(f'Final iter {k}, gap = {gap:.1f}, value = {value.mean():.5f}, ')
        return value, alpha


class GradientOverrider(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f: torch.tensor, grad: torch.tensor):
        ctx.save_for_backward(grad)
        return f

    @staticmethod
    def backward(ctx, grad_output):
        grad, = ctx.saved_tensors
        return grad * grad_output[:, None], None


class Gspace2d(_BaseGspace):
    def __init__(self, h, w, n_channels=1,
                 sigma=1., tol=1e-9,
                 max_iter=1000,
                 verbose=False,
                 method='lbfgs',
                 logspace=True,
                 ):
        """Fast implementation using separable convolution."""

        super().__init__(tol, max_iter, verbose, method)

        self.n_channels = n_channels
        self.h = h
        self.w = w
        self.logspace = logspace

        if self.logspace:
            ch = - torch.arange(0, h, dtype=torch.float) ** 2 / 4 / sigma ** 2
            cw = - torch.arange(0, w, dtype=torch.float) ** 2 / 4 / sigma ** 2

            toeph = torch.empty((h, h))
            toepw = torch.empty((w, w))
            for i in range(h):
                toeph[i, i:] = ch[:h - i]
                toeph[i:, i] = ch[:h - i]
            for i in range(w):
                toepw[i, i:] = cw[:w - i]
                toepw[i:, i] = cw[:w - i]

            self.register_buffer('toeph', toeph)
            self.register_buffer('toepw', toepw)
        else:
            ch = - torch.arange(-h + 1, h, dtype=torch.float32) ** 2 / 4 / sigma ** 2
            cw = - torch.arange(-w + 1, w, dtype=torch.float32) ** 2 / 4 / sigma ** 2
            self.register_buffer('kernel', torch.exp(ch[:, None] + cw[None, :]))

    def _log_conv(self, u: torch.tensor):
        batch_size = u.shape[0]
        if self.logspace:
            u = u.view(batch_size * self.n_channels, self.h, self.w)
            conv_w = u[:, :, :, None] + self.toepw[None, None, :, :]
            conv_w = torch.logsumexp(conv_w, dim=2)
            conv = conv_w[:, :, None, :] + self.toeph[None, :, :, None]
            conv = torch.logsumexp(conv, dim=1)
        else:
            u = u.view(batch_size, self.n_channels, self.h, self.w)
            padding = self.h - 1, self.w - 1
            m = u.view(batch_size, -1).max(dim=1)[0]
            X = torch.exp(u - m[:, None, None, None])
            X = F.conv2d(X, self.kernel[None, None, :, :],
                         padding=padding, groups=self.n_channels)
            X.clamp_(min=1e-18)
            conv = safe_log(X) + m[:, None, None, None]
        return conv.view(batch_size, -1)

    def _get_C(self, s, v):
        ii, jj, kk, ll = s / self.w, s % self.w, v / self.w, v % self.w
        return self.toeph[ii, kk] + self.toepw[jj, ll]


class Gspace1d(_BaseGspace):
    def __init__(self, C, tol=1e-9, max_iter=1000, verbose=False,
                 method='lbfgs'):
        super().__init__(tol, max_iter, verbose, method)
        self.register_buffer('C', - C / 2)

    def _get_C(self, s, v):
        return self.C[s, v]

    def _log_conv(self, u: torch.tensor):
        conv = u[:, None, :] + self.C[None, :, :]
        return torch.logsumexp(conv, dim=2)
