from __future__ import print_function

import os
from os.path import join, expanduser

import joblib
import torch
import torch.utils.data
from sacred import Experiment
from sacred.observers import FileStorageObserver
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset
from torchvision import datasets
from torchvision.transforms import transforms
from torchvision.utils import save_image

from gsoftmax.models import VAE, ConvVAE, LastLayer, WrappedVAE
from gsoftmax.modules import safe_log, Gspace2d

base_dir = expanduser('~/output/ot-entropies')
exp = Experiment('VAE')
exp.observers.append(FileStorageObserver.create(join(base_dir, 'runs')))


class ToProb(object):
    """
    Transform an image onto the simplex.
    """

    def __call__(self, pic):
        """
        """
        n_channel, w, h = pic.shape
        pic -= pic.view(n_channel, -1).min(dim=1)[0][:, None]
        s = torch.sum(pic.view(n_channel, -1), dim=1)
        s[s == 0] = 1
        return pic / s[:, None, None]

    def __repr__(self):
        return self.__class__.__name__ + '()'


@exp.config
def system():
    cuda = True
    device = 0
    output_dir = join(base_dir, 'runs')
    seed = 0
    source = 'mnist'
    checkpoint = False
    log_interval = 100
    supervised_score = False


@exp.config
def base():
    batch_size = 512
    epochs = 100
    loss_type = 'geometric'
    latent_dim = 256
    model_type = 'flat'
    max_iter = 30
    sigma = 1.
    regularization = 1
    model_type = 'conv'
    lr = 1e-3


@exp.capture
def train(model, optimizer, loader, device, log_interval, epoch, _run):
    model.train()
    train_loss = 0
    train_penalty = 0

    for batch_idx, (data, _) in enumerate(loader):
        data = data.to(device)
        optimizer.zero_grad()
        loss, penalty = model(data, return_penalty=True)
        loss.backward()
        train_loss += loss.item()
        train_penalty += penalty.item()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\t'
                'Loss: {:.6f}\tPenalty {:6f}'.format(
                    epoch, batch_idx * len(data), len(loader.dataset),
                           100. * batch_idx / len(loader),
                           loss.item() / len(data),
                           penalty.item() / len(data)))
    train_loss /= len(loader.dataset)
    train_penalty /= len(loader.dataset)
    print(f'====> Epoch: {epoch} Average loss: {train_loss:.4f}'
          f' Average penalty {train_penalty:.4f}')
    _run.log_scalar('train.loss', train_loss, epoch)
    _run.log_scalar('train.penalty', train_penalty, epoch)


@exp.capture
def _test(model, loader, device, epoch, output_dir, gspace, _run):
    model.eval()
    test_loss = 0.
    test_penalty = 0.
    kl_div = 0.
    bregman_div = 0.
    bce = 0.

    with torch.no_grad():
        for i, (data, _) in enumerate(loader):
            batch_size, n_channel, h, w = data.shape
            data = data.to(device)
            loss, penalty = model(data, return_penalty=True)
            test_loss += loss.item()
            test_penalty += penalty.item()
            pred = model.pred(data)

            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], pred[:n]])
                n_batch, n_channel = comparison.shape[:2]
                # Normalize
                comparison /= comparison.view(n_batch, n_channel, -1).max(dim=2)[0][:, :, None, None]
                filename = join(output_dir, 'reconstruction_' + str(epoch) + '.png')
                save_image(comparison.cpu(), filename, nrow=n)
                _run.add_artifact(filename)

            data = data.view(batch_size, -1)
            pred = pred.view(batch_size, -1)
            kl_div += F.kl_div(safe_log(pred), target=data, reduction='sum').item()

            bregman_div += gspace.hausdorff(pred, data, reduction='sum').item()

            bce += F.binary_cross_entropy(pred, target=data,
                                          reduction='sum').item()

    test_loss /= len(loader.dataset)
    test_penalty /= len(loader.dataset)
    kl_div /= len(loader.dataset)
    bregman_div /= len(loader.dataset)
    bce /= len(loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    _run.log_scalar('test.loss', test_loss, epoch)
    _run.log_scalar('test.penalty', test_penalty, epoch)
    _run.log_scalar('test.kl_div', kl_div, epoch)
    _run.log_scalar('test.bregman_div', bregman_div, epoch)
    _run.log_scalar('test.bce', bce, epoch)


@exp.capture
def generate(model, latent_dim, device, output_dir, epoch, _run):
    model.eval()
    with torch.no_grad():
        sample = torch.randn(64, latent_dim).to(device)
        pred = model.decode(sample)
        # Normalize
        pred /= pred.view(
            pred.shape[0], pred.shape[1], -1).max(dim=2)[0][:, :, None, None]
        filename = join(output_dir, 'sample_' + str(epoch) + '.png', )
        save_image(pred, filename)
        _run.add_artifact(filename)


@exp.capture
def save_checkpoint(model, optimizer, output_dir, epoch, _run):
    state_dict = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'epoch': epoch}
    filename = join(output_dir, f'checkpoint_{epoch}.pkl')
    torch.save(state_dict, filename)
    _run.add_artifact(filename)


@exp.automain
def run(device, loss_type, source, cuda, batch_size, checkpoint, output_dir,
        epochs, latent_dim, model_type, max_iter, sigma,
        lr,
        _seed):
    torch.manual_seed(_seed)
    device = torch.device(f"cuda:{device}" if cuda else "cpu")

    if loss_type == 'bce':
        transform = transforms.ToTensor()
    elif loss_type == ['kl', 'metric']:
        transform = transforms.Compose([transforms.ToTensor(), ToProb()])
    else:
        raise ValueError('Wrong `loss_type` argument')

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    if source == 'mnist':
        train_data = datasets.MNIST('../data', train=True, download=True,
                                    transform=transform)
        test_data = datasets.MNIST('../data', train=False, transform=transform)
        h, w = 28, 28
    elif 'quickdraw' in source:
        _, class_name, size = source.split('_')
        size = int(size)
        data_dir = join(expanduser('~/data/quickdraw/bitmaps'))
        h = w = size
        data = {}
        for fold in ['train', 'test']:
            filename = join(data_dir, f'{class_name}_{size}_{fold}.pkl')
            x, y = joblib.load(filename)
            x, y = torch.from_numpy(x), torch.from_numpy(y)
            x = x.view(x.shape[0], 1, h, w)
            data[fold] = TensorDataset(x, y)
        test_data, train_data = data['test'], data['train']
    else:
        raise ValueError('Wrong `source` argument')

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=batch_size,
                                              shuffle=True, **kwargs)
    if model_type == 'conv':
        vae = ConvVAE(h, w, latent_dim).to(device)
    else:
        vae = VAE(h, w, latent_dim).to(device)

    gspace = Gspace2d(h, w, sigma=sigma, tol=1e-4, max_iter=max_iter, verbose=True)
    last_layer = LastLayer(loss_type=loss_type, gspace=gspace)

    model = WrappedVAE(vae, last_layer)

    model = model.to(device)

    optimizer = optim.Adam(vae.parameters(), lr=lr)

    if checkpoint:
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        start_epoch = state_dict['epoch']
    else:
        start_epoch = 0

    output_dir = join(output_dir, str(exp._id), 'artifacts')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for epoch in range(start_epoch, epochs + 1):
        train(model=model, optimizer=optimizer, loader=train_loader,
              epoch=epoch, device=device)
        _test(model=model, loader=test_loader, epoch=epoch, device=device,
              gspace=gspace,  # To track Hausdorff distance
              output_dir=output_dir)
        generate(model=model, epoch=epoch, device=device,
                 output_dir=output_dir)
        if epoch % 10 == 0:
            save_checkpoint(model=vae, optimizer=optimizer, epoch=epoch,
                            output_dir=output_dir)
