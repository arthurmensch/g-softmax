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


class _BaseGSpaceImpl:
    def __init__(self, tol=1e-9,
                 max_iter=1000,
                 verbose=False,
                 method='lbfgs',
                 ):
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

        self.method = method

    def log_conv(self, u: torch.tensor):
        raise NotImplementedError

    def get_C(self, s, v):
        raise NotImplementedError

    def to(self, device):
        return self

    def c_transform(self, f: torch.tensor, log_alpha: torch.tensor):
        f = f / 2 + log_alpha
        log_conv = self.log_conv(f)
        return - 2 * log_conv

    def entropy(self, alpha: torch.tensor):
        f = torch.zeros_like(alpha)
        log_alpha = safe_log(alpha)
        for i in range(self.max_iter):
            g = self.c_transform(f, log_alpha)
            diff = f - g
            gap = (torch.max(diff, dim=1)[0] - torch.min(diff, dim=1)[0]).mean()
            if self.verbose:
                print(f'[Negentropy] Iter {i}, gap {gap}')
            if gap.mean() < self.tol:
                break
            f = .5 * (f + g)
        f = -f
        v = (f * alpha).sum(dim=1)
        return v, f

    def lse(self, f: torch.tensor, init=None):
        if self.method == 'lbfgs':
            return self._lse_lbfgs(f, init)
        elif self.method in ['fw', 'fw_ls', 'fw_away', 'fw_pw']:
            return self._lse_fw(f, init)
        else:
            raise ValueError('Wrong method')

    def _quadratic(self, l, f):
        batch_size, _ = l.shape
        lse = torch.logsumexp(l, dim=1)
        l = l - lse[:, None] + f
        log_quad = l + self.log_conv(l)
        log_quad = log_quad.view(batch_size, -1)
        return - torch.logsumexp(log_quad, dim=1), log_quad

    def _lse_lbfgs(self, f: torch.tensor, init=None):
        batch_size, _ = f.shape

        if init is not None:
            init = init.view(batch_size, -1)
            l = safe_log(init.clone())
        else:
            l = f.clone()
        l -= torch.logsumexp(l, dim=1)[:, None]

        f = -f / 2

        v_init, _ = self._quadratic(l, f)
        v_init = v_init.mean().item()

        if self.max_iter > 0:
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
                    v, _ = self._quadratic(l, f)
                    v = - v.sum()
                    if self.verbose > 10:
                        self.n_eval += 1
                        print(f'Eval {self.n_eval} value {v.item()}')
                    v.backward()
                    return v

                optimizer.step(closure)
            l = l.detach()
            l -= torch.logsumexp(l, dim=1)[:, None]
            v, log_g = self._quadratic(l, f)
            if self.verbose:
                print(
                    f'Initial quadratic value: {v_init:.3f}, maximized value: {v.mean().item():.3f}')

        return v, torch.softmax(l, dim=1)

    def _lse_fw(self, f: torch.tensor, init=None):
        batch_size, _ = f.shape

        if init is not None:
            init = init.view(batch_size, -1)
            alpha = init.clone()
        else:
            alpha = F.softmax(f, dim=1)

        f = -f.view(batch_size, -1) / 2

        for k in range(self.max_iter):
            lgrad = f + self.log_conv(f + safe_log(alpha))
            laa = torch.logsumexp(safe_log(alpha) + lgrad, dim=1)

            s = torch.min(lgrad, dim=1)[1]
            lsa = lgrad[range(batch_size), s]

            value = - laa
            m = torch.max(laa, lsa)
            log_gap = m + torch.log(torch.exp(laa - m) - torch.exp(lsa - m))

            if self.method in ['fw_away', 'fw_pw']:
                lgrad_masked = lgrad.clone()
                lgrad_masked[alpha <= 0] = - float('inf')
                v = torch.max(lgrad_masked, dim=1)[1]
                lva = lgrad[range(batch_size), v]
                m = torch.max(laa, lva)
                log_gap_away = m + torch.log(
                    torch.exp(lva - m) - torch.exp(laa - m))
            if self.method == 'fw_pw':
                lss = 2 * f[range(batch_size), s]
                lvv = 2 * f[range(batch_size), v]

                max_step_size = alpha[range(batch_size), v]

                lsv = f[range(batch_size), s] + f[range(batch_size), v]
                lsv += self.get_C(s)
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
                    lss = 2 * f[range(batch_size), s]
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

            step_size = step_size if self.method == "fw" else step_size.mean()
            log_gap = torch.exp(log_gap).mean()

            if self.verbose and k % 1 == 0:
                info = (f'Iter {k}, gap = {log_gap:.1f}, '
                        f"value = {value.mean():.5f}, "
                        f's = {s}, '
                        f'step_size = {step_size:.2e}')
                if self.method in ['fw_pw', 'fw_away']:
                    gap_away = torch.exp(log_gap_away).mean()
                    info += f', gap_away = {gap_away:.1f}'
                if self.method == 'fw_pw':
                    info += f', v = {v}'
                print(info)
            if log_gap < self.tol or step_size == 0.:
                break

        return value, alpha


class _GSpace2dImpl(_BaseGSpaceImpl):
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

        ch = - torch.tensor(list(range(-h, h + 1)),
                            dtype=torch.float) ** 2 / 4 / sigma ** 2
        cw = - torch.tensor(list(range(-w, w + 1)),
                            dtype=torch.float) ** 2 / 4 / sigma ** 2

        if self.logspace:
            self.toeph = torch.zeros((h, h))
            self.toepw = torch.zeros((w, w))
            for i in range(h):
                self.toeph[i, :] = ch[h - i:2 * h - i]
            for i in range(w):
                self.toepw[i, :] = cw[w - i:2 * w - i]
            else:
                self.kernel = ch[h - 1:h + 2][:, None] + cw[w - 1:w + 2][None, :]

    def to(self, device):
        if self.logspace:
            self.toepw = self.toepw.to(device)
            self.toeph = self.toeph.to(device)
        else:
            self.kernel = self.kernel.to(device)
        return self

    def log_conv(self, u: torch.tensor):
        batch_size = u.shape[0]
        u = u.view(batch_size * self.n_channels, self.h, self.w)
        if self.logspace:
            conv_w = u[:, :, :, None] + self.toepw[None, None, :, :]
            conv_w = torch.logsumexp(conv_w, dim=2)
            conv = conv_w[:, :, None, :] + self.toeph[None, :, :, None]
            conv = torch.logsumexp(conv, dim=1)
        else:
            batch_size, h, w = u.shape
            padding = (self.kernel.shape[2] - 1) // 2, (
                    self.kernel.shape[3] - 1) // 2
            m = u.view(batch_size, -1).max(dim=1)[0]
            X = torch.exp(u - m[:, None, None])
            X = F.conv2d(X[:, None], self.kernel, padding=padding)[:, 0]
            X.clamp_(min=1e-15)
            conv = safe_log(X) + m[:, None, None]
        return conv.view(-1, self.in_channels, self.h, self.w)

    def get_C(self, s, v):
        ii, jj, kk, ll = s / self.w, s % self.w, v / self.w, v % self.w
        return self.toeph[ii, kk] + self.toepw[jj, ll]


class _GSpace1dImpl(_BaseGSpaceImpl):
    def __init__(self, C, tol=1e-9,
                 max_iter=1000,
                 verbose=False,
                 method='lbfgs',
                 ):
        super().__init__(tol, max_iter, verbose, method)
        self.C = C / 2

    def to(self, device):
        self.C = self.C.to(device)

    def get_C(self, s, v):
        return self.C[s, v]

    def log_conv(self, u: torch.tensor):
        conv = u[:, :, None] + self.C[None, :, :]
        return torch.logsumexp(conv, dim=2)


class gLSEFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f: torch.tensor, gspace: _BaseGSpaceImpl):
        with torch.no_grad():
            value, grad = gspace.lse(f)
        ctx.save_for_backward(grad)
        return value

    @staticmethod
    def backward(ctx, grad_output):
        grad, = ctx.saved_tensors
        return grad * grad_output[:, None, None], None


class gSoftmaxFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f: torch.tensor, gspace: _BaseGSpaceImpl):
        with torch.no_grad():
            value, grad = gspace.lse(f)
        return grad

    @staticmethod
    def backward(ctx, grad_output):
        return None, None


class gEntropyFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f: torch.tensor, gspace: _BaseGSpaceImpl):
        with torch.no_grad():
            value, grad = gspace.entropy(f)
        ctx.save_for_backward(grad)
        return value

    @staticmethod
    def backward(ctx, grad_output):
        grad, = ctx.saved_tensors
        return grad * grad_output[:, None, None], None


class gPotentialFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f: torch.tensor, gspace: _BaseGSpaceImpl):
        with torch.no_grad():
            value, grad = gspace.entropy(f)
        return grad

    @staticmethod
    def backward(ctx, grad_output):
        return None, None


