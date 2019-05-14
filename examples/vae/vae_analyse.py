from os.path import expanduser, join

import joblib
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torchvision.utils import save_image

from gsoftmax.models import VAE
from gsoftmax.modules import Gspace2d

image = True
curves = True

if image:
    torch.manual_seed(200)

    # kl = expanduser('~/output/ot-entropies/results/kl_small_sigma_2')
    # metric = expanduser('~/output/ot-entropies/results/metric_small_sigma_2')
    # data_dir = join(expanduser('~/data/quickdraw/numpy_bitmap/derivatives'))

    kl = expanduser('~/output/g-softmax/vae/2')
    metric = expanduser('~/output/g-softmax/vae/4')
    data_dir = join(expanduser('~/data/quickdraw/numpy_bitmap/derivatives'))

    source = 'quickdraw_ambulance_64'

    data_dir = join(expanduser('~/data/quickdraw/bitmaps'))
    _, class_name, size = source.split('_')
    size = int(size)
    h = w = size
    data = {}
    for fold in ['train', 'test']:
        filename = join(data_dir,
                        f'{class_name}_{size}_{fold}'
                        f'_norm.pkl')
        x, y = joblib.load(filename)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        x = x.view(x.shape[0], 1, h, w)
        data[fold] = TensorDataset(x, y)

    sigma = 1
    max_iter = 10
    latent_dim = 256
    device = 0
    model_type = 'conv'
    regularization = 0.01
    h, w = size, size
    cuda = True

    test_data = data['test']

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=8,
                                              shuffle=True, **kwargs)
    data, _ = next(iter(test_loader))
    data = data.to(device)

    checkpoints = {'kl': join(kl, 'artifacts', 'checkpoint_20.pkl'),
                   'geometric': join(metric, 'artifacts', 'checkpoint_20.pkl')}

    z = torch.randn(25, latent_dim).to(device)

    recs = {}

    metric_softmax = Gspace2d(h, w, sigma=sigma, tol=1e-4, max_iter=max_iter,
                              method='lbfgs', verbose=False)

    for loss_type, checkpoint in checkpoints.items():
        model = VAE(h, w, latent_dim, loss_type=loss_type, model_type=model_type,
                    gspace=metric_softmax, regularization=regularization).to(device)
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict['model'])
        model.eval()

        with torch.no_grad():
            recs[loss_type] = model.pred(data)
            pred = model.pred_from_latent(z)
            pred /= pred.view(pred.shape[0], 1, -1).max(dim=2)[0][:, :, None, None]

            save_image(pred, expanduser(f'~/output/g-softmax/sample_{loss_type}.png'), nrow=5)

    all_recs = torch.zeros(24, 1, h, w)
    all_recs[:8] = data[:8]
    all_recs[8:16] = recs['kl']
    all_recs[16:24] = recs['geometric']

    all_recs /= all_recs.view(all_recs.shape[0], 1, -1).max(dim=2)[0][:, :, None, None]
    save_image(all_recs, expanduser(f'~/output/g-softmax/reconstruction.png'), nrow=8)

##########

if curves:
    import json

    with open(join(kl, 'metrics.json'), 'r') as f:
        m_kl = json.load(f)

    with open(join(metric, 'metrics.json'), 'r') as f:
        m_metric = json.load(f)

    from matplotlib import rc

    rc('font', **{'family': 'sans-serif'})
    # rc('text', usetex=True)
    # rc('text.latex', preamble=r'\usepackage{amsmath}')
    # rc('figure.constrained_layout', use=True)
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(1, 2, figsize=(3.2, 1.5),
                             gridspec_kw=dict(width_ratios=[2, .7]),
                             )
    fig.subplots_adjust(bottom=0.3, left=0.2, right=0.95, wspace=0.4, top=0.95)

    h1, = axes[0].plot(m_kl['test.bregman_div']['steps'], m_kl['test.bregman_div']['values'],
                       marker='.', markevery=5, markersize=1, color='C1')
    axes[0].plot(m_metric['test.bregman_div']['steps'], m_metric['test.bregman_div']['values'], marker='.',
                 markevery=5,
                 markersize=2, color='C0')
    print(m_kl['test.kl_div']['values'])
    h2, = axes[0].plot(m_kl['test.kl_div']['steps'], np.array(m_kl['test.kl_div']['values']) - 0.3, marker='.',
                       linestyle=':',
                       markevery=5,
                       markersize=1, color='C1', label='$\D_\Omega$')
    axes[0].plot(m_metric['test.kl_div']['steps'][::5], np.array(m_metric['test.kl_div']['values'][::5]) - 0.3,
                 marker='.',
                 linestyle=':', markersize=1,
                 color='C0')
    leg = axes[0].legend([h1, h2], ['$D_\Omega$', 'KL div.'], loc='upper right', ncol=2, frameon=False,
                         columnspacing=0.5,
                         handlelength=.5)
    # axes[0].annotate('$D_\Omega$', fontsize=13, color='C0', xy=(25, 0.2), xycoords='data')
    # axes[0].annotate('K-L', fontsize=13, color='C1', xy=(75, 0.5), xycoords='data')

    axes[0].set_ylim([0, 2])

    axes[1].set_ylabel('Latent K-L')
    axes[0].annotate('Epoch', xy=(-.4, -.2), xycoords='axes fraction')
    axes[0].set_ylabel('Loss')

    axes[1].plot(m_kl['test.penalty']['steps'], m_kl['test.penalty']['values'], color='C1',
                 label='softmax')
    axes[1].plot(m_metric['test.penalty']['steps'], m_metric['test.penalty']['values'], color='C0',
                 label='g-softmax')
    axes[1].tick_params(axis='y', which='major', labelsize=7)
    axes[1].legend(loc='upper right', bbox_to_anchor=(-.6, -.15), ncol=2, frameon=False, mode='expand',
                   handlelength=1, columnspacing=0.5)

    sns.despine(fig)
    plt.savefig(expanduser(f'~/output/g-softmax/quantitative.png'))
    # plt.show()
