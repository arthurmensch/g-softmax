import pytest
import torch
from numpy.testing import assert_array_almost_equal

from affinity import safe_log
from gsoftmax.euclidean import _BaseGSpaceImpl


@pytest.mark.parametrize('device', ['cpu'])
@pytest.mark.parametrize('method', ['lbfgs'])
def test_metric_softmax(device, method):
    if 'cuda' in device and not torch.cuda.is_available():
        pytest.skip('Not GPU found, skipping')
    torch.set_num_threads(1)
    h, w = 40, 40
    batch_size = 3
    alpha = torch.zeros((batch_size, h, w))
    alpha[:, 10, 10] = 1
    alpha[:, 10, 20] = 1
    alpha[:, 10, 5] = 1
    alpha[:, 15, 5] = .1
    alpha[:, 28, 38] = 1
    alpha /= alpha.view(batch_size, -1).sum(dim=1)[:, None, None]

    metric_space = _BaseGSpaceImpl(h, w, verbose=True, max_iter=1000,
                                   sigma=1, epsilon=2, logspace=False,
                                   method=method)

    alpha = alpha.to(device)
    metric_space = metric_space.to(device)

    _, f = metric_space.entropy(alpha)
    g = - metric_space.c_transform(-f, safe_log(alpha))
    assert_array_almost_equal(f.cpu().numpy(), g.cpu().numpy(), 3)

    _, pred = metric_space.lse(f)

    pred = pred.to('cpu').numpy()
    alpha = alpha.to('cpu').numpy()
    assert_array_almost_equal(pred, alpha, 2)
