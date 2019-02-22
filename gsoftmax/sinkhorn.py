import torch

def bmm(x: torch.tensor, y):
    """
    Batched matrix multiplication

    :param x:
    :param y:
    :return:
    """
    return torch.einsum('blk,bk->bl', x, y)


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


def pairwise_distance(x, y, q=2, p=2, sigma=1, exp=False, ):
    if p == 2:
        outer = torch.einsum('bld,bkd->blk', x, y)
        norm_x = torch.sum(x ** 2, dim=2)
        norm_y = torch.sum(y ** 2, dim=2)
        distance = ((norm_x[:, :, None] + norm_y[:, None, :]
                     - 2 * outer))
        distance.clamp_(min=0.)
        if q == 1:
            k = - torch.sqrt(distance + 1e-8) / sigma
        elif q == 2:
            k = - distance / 2 / sigma ** 2
        else:
            raise ValueError(f'Wrong q argument for p==2, got `{q}`')
    elif p == 1:
        diff = x[:, :, None, :] - y[:, None, :, :]
        distance = (torch.sum(torch.abs(diff), dim=3))
        if q == 1:
            k = - distance / sigma
        else:
            raise ValueError(f'Wrong q argument for p==1, got `{q}`')
    else:
        raise ValueError(f'Wrong `p` argument, got `{p}`')
    if exp:
        k = torch.exp(k)
    return k


def c_transform(potential, log_weight, kernel, epsilon, rho):
    # Edge case potential = + infty, weight = - infty
    scale = 1 if rho is None else 1 + epsilon / rho
    # potential = masker(potential, log_weight == -float('inf'), -float('inf'))
    sum = potential / epsilon + log_weight
    operand = kernel / epsilon + sum[:, None, :]
    # lse = - epsilon * logsumexp_inf(operand)
    lse = - epsilon * torch.logsumexp(operand, dim=2)
    return lse / scale


def phi_transform(f, epsilon, rho):
    if rho is None:
        return f
    else:
        scale = 1 + epsilon / rho
        # f = f.clone()
        # mask = torch.isfinite(f)
        # f[mask] = - rho * ((- f[mask] / rho).exp() - 1)
        f = - rho * ((-f / rho).exp() - 1)
        return f * scale


def sym_potential(x, a, p, q, sigma, epsilon, rho, max_iter, tol):
    kxx = pairwise_distance(x, x.detach(), p, q, sigma)
    log_a = torch.log(a)

    f = torch.zeros_like(log_a)
    for n_iter in range(max_iter):
        fn = c_transform(f, log_a, kxx, epsilon, rho)
        gap = duality_gap(fn, f)
        f = (fn + f) / 2
        if gap < tol:
            break
    return f


def potentials(x, a, y, b, p, q, sigma, epsilon, rho, max_iter, tol):
    kxy = pairwise_distance(x, y, p, q, sigma)
    kyx = kxy.transpose(1, 2)
    log_a, log_b = torch.log(a), torch.log(b)
    f, g = torch.zeros_like(log_a), torch.zeros_like(log_b)
    for n_iter in range(max_iter):
        fn = c_transform(g, log_b, kxy, epsilon, rho)
        gn = c_transform(f, log_a, kyx, epsilon, rho)
        gap = duality_gap(fn, f) + duality_gap(gn, g)
        f, g = (fn + f) / 2, (gn + g) / 2
        if gap < tol:
            break
    return f, g


def evaluate_potential(x, a, y, b, g, p, q, sigma, epsilon, rho):
    y = y.detach()
    b = b.detach()
    log_b = b.log()
    kxy = pairwise_distance(x, y, p, q, sigma)

    f = c_transform(g, log_b, kxy, epsilon, rho)
    f = phi_transform(f, epsilon, rho)
    # f = masker(f, a == 0, 0)
    return scaled_dot_prod(a, f)
    # return torch.sum(a * f, dim=1)


def sinkhorn_distance(x, a, y, b, p, q, sigma, epsilon, rho, max_iter, tol):
    f, g = potentials(x, a, y, b, p, q, sigma, epsilon, rho, max_iter, tol)
    f, g = f.detach(), g.detach()
    return (evaluate_potential(x, a, y, b, g, p, q, sigma, epsilon, rho)
            + evaluate_potential(y, b, x, a, f, p, q, sigma, epsilon, rho))


def sinkhorn_entropy(x, a, p, q, sigma, epsilon, rho, max_iter, tol):
    f = sym_potential(x, a, p, q, sigma, epsilon, rho, max_iter, tol)
    f = f.detach()
    return evaluate_potential(x, a, x, a, f, p, q, sigma, epsilon, rho)


def sinkhorn_divergence(x, a, y, b, p, q, sigma, epsilon, rho, max_iter, tol):
    return (sinkhorn_distance(x, a, y, b, p, q, sigma, epsilon,
                              rho, max_iter, tol)
            - sinkhorn_entropy(x, a, p, q, sigma, epsilon, rho, max_iter, tol)
            - sinkhorn_entropy(y, b, p, q, sigma, epsilon, rho, max_iter, tol))


def hausdorff_divergence(x, a, y, b, p, q, sigma,
                         epsilon, rho, max_iter, tol):
    f = sym_potential(x, a, p, q, sigma, epsilon, rho, max_iter, tol)
    return (evaluate_potential(y, b, x, a, f, p, q, sigma, epsilon, rho)
            - sinkhorn_entropy(y, b, p, q, sigma, epsilon, rho, max_iter, tol))


def rev_hausdorff_divergence(x, a, y, b, p, q, sigma,
                         epsilon, rho, max_iter, tol):
    return hausdorff_divergence(y, b, x, a, p, q, sigma,
                         epsilon, rho, max_iter, tol)


def sym_hausdorff_divergence(x, a, y, b, p, q, sigma,
                             epsilon, rho, max_iter, tol):
    return (hausdorff_divergence(x, a, y, b, p, q, sigma,
                                 epsilon, rho, max_iter, tol) +
            hausdorff_divergence(y, b, x, a, p, q, sigma,
                                 epsilon, rho, max_iter, tol)) / 2


class LogSumExpInf(torch.autograd.Function):
    @staticmethod
    def forward(ctx, operand):
        ctx.save_for_backward(operand)
        return torch.logsumexp(operand, dim=2)

    @staticmethod
    def backward(ctx, grad_output):
        operand, = ctx.saved_tensors
        mask = torch.all(torch.isinf(operand), dim=2) ^ 1
        s = torch.ones_like(operand) / operand.shape[2]
        s[mask] = torch.softmax(operand[mask], dim=1)
        return s * grad_output[:, :, None]


logsumexp_inf = LogSumExpInf.apply


class Masker(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask, target):
        x = x.clone()
        x[mask] = target
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


masker = Masker.apply


class ScaledDotProd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, f):
        ctx.save_for_backward(a, f)
        return torch.sum(a * f, dim=1)

    @staticmethod
    def backward(ctx, grad_output):
        a, f = ctx.saved_tensors
        return (f * grad_output[:, None],
                torch.ones_like(a) * grad_output[:, None])


scaled_dot_prod = ScaledDotProd.apply


class SinkhornDivergence(torch.nn.Module):
    def __init__(self, p, q, sigma, epsilon, rho, max_iter, tol):
        super().__init__()
        self.p = p
        self.q = q
        self.sigma = sigma
        self.epsilon = epsilon
        self.rho = rho
        self.max_iter = max_iter
        self.tol = tol

    def forward(self, x, a, y, b):
        return sinkhorn_divergence(x, a, y, b, self.p, self.q,
                                   self.sigma, self.epsilon, self.rho,
                                   self.max_iter, self.tol).mean()
