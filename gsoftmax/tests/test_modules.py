import pytest
import torch
from numpy.testing import assert_array_almost_equal
from torch.nn import Parameter

from gsoftmax.modules import Gspace2d, safe_log


@pytest.mark.parametrize('device', ['cpu'])
@pytest.mark.parametrize('method', ['fw', 'lbfgs'])
def test_gspace_2d(device, method):
    if 'cuda' in device and not torch.cuda.is_available():
        pytest.skip('Not GPU found, skipping')
    torch.set_num_threads(1)
    h, w, n_channels = 10, 10, 1
    batch_size = 3
    alpha = torch.zeros((batch_size, n_channels, h, w))
    alpha[:, :, 5, 5] = 1
    alpha /= alpha.view(batch_size * n_channels, -1).sum(dim=1).view(
        batch_size,
        n_channels, 1, 1)

    gspace = Gspace2d(h, w, verbose=True, max_iter=1000,
                      sigma=1, logspace=True,
                      method=method)

    alpha = alpha.to(device)
    metric_space = gspace.to(device)

    alpha = alpha.view(batch_size, -1)

    f = metric_space.potential(alpha)
    g = -metric_space.c_transform(-f, safe_log(alpha))
    assert_array_almost_equal(f.cpu().numpy(), g.cpu().numpy(), 3)

    pred = metric_space.softmax(f)

    pred = pred.to('cpu').numpy()
    alpha = alpha.to('cpu').numpy()
    assert_array_almost_equal(pred, alpha, 2)


@pytest.mark.parametrize('device', ['cpu'])
def test_gspace_double_backward(device):
    if 'cuda' in device and not torch.cuda.is_available():
        pytest.skip('Not GPU found, skipping')
    torch.set_num_threads(1)
    h, w, n_channels = 10, 10, 1
    batch_size = 3
    alpha = torch.zeros((batch_size, n_channels, h, w))
    alpha[:, :, 5, 5] = 1
    alpha /= alpha.view(batch_size * n_channels, -1).sum(dim=1).view(
        batch_size,
        n_channels, 1, 1)

    gspace = Gspace2d(h, w, verbose=True, max_iter=1000,
                      sigma=1, logspace=True,
                      create_graph=True,
                      method='lbfgs')

    alpha = alpha.to(device)
    alpha = Parameter(alpha)

    metric_space = gspace.to(device)

    alpha = alpha.view(batch_size, -1)

    f = metric_space.potential(alpha)

    v = f.sum()
    v.backward()
    print(alpha.grad)
