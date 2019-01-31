import pytest
import torch
from numpy.testing import assert_array_almost_equal
from sklearn.utils import check_random_state
import numpy as np

from gsoftmax.entropies import sink_negentropy, conj_sink_negentropy
from gsoftmax.functional import c_transform, safe_log


def make_C():
    c12, c13, c23 = 1, 1, 10

    C = np.array([[0., c12, c13], [0., 0., c23], [0., 0., 0.]])
    C = C + C.T
    return C


def verify_C(C, epsilon):
    eig, _ = np.linalg.eig(np.exp(-C / epsilon))
    assert np.all(eig >= 1e-16), ValueError(
        f'exp(-C) is not positive definite. Spectrum {eig}')


@pytest.mark.parametrize("epsilon", [.1, 1, 10],
                         ids=lambda eps: f'eps={eps:.0e}')
def test_self_sinkhorn(epsilon):
    random_state = check_random_state(0)
    alpha = random_state.rand(10, 3)
    alpha /= alpha.sum(axis=1, keepdims=True)
    C = make_C()

    alpha = torch.from_numpy(alpha)
    C = torch.from_numpy(C)

    v, f = sink_negentropy(C, alpha, epsilon=epsilon, verbose=True)
    f = -f
    g = c_transform(C, f, safe_log(alpha), epsilon=epsilon)

    g = g.numpy()
    f = f.numpy()

    assert_array_almost_equal(g, f)


@pytest.mark.parametrize("epsilon", [.1, 1, 2],
                         ids=lambda eps: f'eps={eps:.0e}')
@pytest.mark.parametrize("method", ['fw', 'lbfgs'])
def test_self_sinkhorn_conj_inverse(epsilon, method):
    random_state = check_random_state(0)
    alpha = random_state.rand(10, 3)
    alpha[0] = 0
    alpha[0, 1] = 1
    alpha /= alpha.sum(axis=1, keepdims=True)
    C = make_C()

    eig, _ = np.linalg.eig(np.exp(-C / epsilon))
    assert np.all(eig >= 1e-16), ValueError(
        f'exp(-C) is not positive definite. Spectrum {eig}')

    alpha = torch.from_numpy(alpha)
    C = torch.from_numpy(C)

    v, f = sink_negentropy(C, alpha, epsilon=epsilon, verbose=True)

    vconj, proj = conj_sink_negentropy(C, f, epsilon=epsilon, verbose=True,
                                       method=method)


    alpha = alpha.numpy()
    proj = proj.numpy()

    assert_array_almost_equal(vconj, 0)
    assert_array_almost_equal(proj, alpha)
