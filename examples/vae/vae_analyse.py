import copy
from os.path import expanduser, join

import joblib
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torchvision.utils import save_image

import torch

from gsoftmax.euclidean import _BaseGSpaceImpl
from vae_train import VAE

import numpy as np

image = False
curves = True

if image:
    torch.manual_seed(200)

    kl = expanduser('~/output/ot-entropies/results/kl_small_sigma_2')
    metric = expanduser('~/output/ot-entropies/results/metric_small_sigma_2')

    data_dir = join(expanduser('~/data/quickdraw/numpy_bitmap/derivatives'))
    #
    sigma = 2
    max_iter = 20
    device = 0
    h, w = 28, 28
    cuda = True

    data = {}
    for fold in ['test']:
        filename = join(data_dir, f'cat_{fold}.pkl')
        x, y = joblib.load(filename)
        x, y = torch.from_numpy(x), torch.from_numpy(y)

        x /= torch.sum(x, dim=1)[:, None]
        x = x.view(x.shape[0], 1, h, w)

        data[fold] = TensorDataset(x, y)

    test_data = data['test']

    metric_softmax = _BaseGSpaceImpl(h, w, epsilon=2, sigma=sigma, tol=1e-5,
                                     max_iter=max_iter, verbose=False).to(device)

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=512,
                                              shuffle=True, **kwargs)

    model = VAE(h, w, 100).to(device)

    checkpoints = {'kl': join(kl, 'checkpoint_100.pkl'),
                   'metric': join(metric, 'checkpoint_100.pkl')}

    sample = torch.randn(25, 100).to(device)
    data, _ = next(iter(test_loader))
    data = data.to(device)

    preds = {}

    for loss_type, checkpoint in checkpoints.items():
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict['model'])
        model.eval()

        with torch.no_grad():
            logits = model.decode(sample)
            logits_rec, _, _ = model(data)

            if loss_type == 'kl':
                preds[loss_type] = F.softmax(logits_rec.view(-1, h * w),
                                        dim=1).view(512, 1, h, w)[:8]
            else:
                preds[loss_type] = metric_softmax(logits_rec.view(-1, h, w)).view(512, 1, h, w)[:8]

            for last in ['kl', 'metric']:
                if last == 'kl':
                    pred = F.softmax(logits.view(-1, h * w),
                                        dim=1).view(25, 1, h, w)
                else:
                    pred = metric_softmax(logits.view(-1, h, w)).view(25, 1, h, w)

                pred /= pred.view(pred.shape[0], 1, -1).max(dim=2)[0][:, :, None, None]

                save_image(pred, expanduser(f'~/output/analyse/sample_{loss_type}_{last}.png'), nrow=5)

    pred = torch.zeros(24, 1, h, w)
    pred[:8] = data[:8]
    pred[8:16] = preds['kl']
    pred[16:24] = preds['metric']

    pred /= pred.view(pred.shape[0], 1, -1).max(dim=2)[0][:, :, None, None]
    save_image(pred, expanduser(f'~/output/analyse/reconstruction.png'), nrow=8)

##########

if curves:

    import json

    with open(expanduser('~/output/ot-entropies/runs/35/metrics.json'), 'r') as f:
        m_kl = json.load(f)

    with open(expanduser('~/output/ot-entropies/runs/32/metrics.json'), 'r') as f:
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


    h1, =axes[0].plot(m_kl['test.bregman_div']['steps'], m_kl['test.bregman_div']['values'],
                 marker='.', markevery=5, markersize=1, color='C1')
    axes[0].plot(m_metric['test.bregman_div']['steps'], m_metric['test.bregman_div']['values'], marker='.',
                 markevery=5,
                 markersize=2, color='C0')
    print(m_kl['test.kl_div']['values'])
    h2,= axes[0].plot(m_kl['test.kl_div']['steps'], np.array(m_kl['test.kl_div']['values']) - 0.3, marker='.', linestyle=':',
                 markevery=5,
                 markersize=1, color='C1', label='$\D_\Omega$')
    axes[0].plot(m_metric['test.kl_div']['steps'][::5], np.array(m_metric['test.kl_div']['values'][::5]) - 0.3, marker='.',
                     linestyle=':', markersize=1,
                     color='C0')
    leg = axes[0].legend([h1, h2], ['$D_\Omega$', 'KL div.'], loc='upper right', ncol=2, frameon=False, columnspacing=0.5,
                         handlelength=.5)
    # axes[0].annotate('$D_\Omega$', fontsize=13, color='C0', xy=(25, 0.2), xycoords='data')
    # axes[0].annotate('K-L', fontsize=13, color='C1', xy=(75, 0.5), xycoords='data')

    axes[0].set_ylim([0, 0.2])

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
    plt.savefig('vae_quantitative.pdf')
    plt.show()