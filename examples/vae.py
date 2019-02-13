import os
from os.path import join, expanduser

import joblib
import torch
import torch.utils.data

from sacred import SETTINGS

SETTINGS.HOST_INFO.INCLUDE_GPU_INFO = False

from sacred import Experiment
from sacred.observers import FileStorageObserver
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset
from torchvision import datasets
from torchvision.transforms import transforms
from torchvision.utils import save_image

from gsoftmax.models import VAE
from gsoftmax.modules import safe_log, Gspace2d

base_dir = expanduser('~/output/g-softmax/vae')
exp = Experiment('vae')
exp.observers.append(FileStorageObserver.create(base_dir))


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
    seed = 0
    source = 'quickdraw_ambulance_64'
    checkpoint = False
    log_interval = 10
    supervised_score = False


@exp.config
def base():
    batch_size = 512
    epochs = 100
    loss_type = 'geometric'
    latent_dim = 256
    model_type = 'conv'
    max_iter = 5
    sigma = 3.
    regularization = .01
    lr = 1e-3
    # adversarial specific
    prob_param = 'sigmoid'
    gradient_reversal = False


@exp.capture
def train(model, optimizer, reverse_optimizer,
          loader, device, log_interval, epoch, _run):
    model.train()
    train_loss = 0
    train_penalty = 0

    for batch_idx, (data, _) in enumerate(loader):
        data = data.to(device)

        if reverse_optimizer is not None and batch_idx % 5 == 0:
            reverse_optimizer.zero_grad()
            loss, penalty = model(data, return_penalty=True)
            (- loss).backward()
            reverse_optimizer.step()

        optimizer.zero_grad()
        loss, penalty = model(data, return_penalty=True)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_penalty += penalty.item()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
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
                comparison /= \
                    comparison.view(n_batch, n_channel, -1).max(dim=2)[0][:, :,
                    None, None]
                filename = join(output_dir,
                                'reconstruction_' + str(epoch) + '.png')
                save_image(comparison.cpu(), filename, nrow=n)
                # _run.add_artifact(filename)

            data = data.view(batch_size, -1)
            pred = pred.view(batch_size, -1)
            kl_div += F.kl_div(safe_log(pred), target=data,
                               reduction='sum').item()

            if gspace is not None:
                bregman_div += gspace.hausdorff(pred, data,
                                                reduction='sum').item()

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
        pred = model.pred_from_latent(sample)
        # Normalize
        pred /= pred.view(
            pred.shape[0], pred.shape[1], -1).max(dim=2)[0][:, :, None, None]
        filename = join(output_dir, 'sample_' + str(epoch) + '.png', )
        save_image(pred, filename)
        # _run.add_artifact(filename)


@exp.capture
def save_checkpoint(model, optimizer, reverse_optimizer,
                    output_dir, epoch, _run):
    state_dict = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'reverse_optimizer': reverse_optimizer.state_dict() if reverse_optimizer is not None else None,
                  'epoch': epoch}
    filename = join(output_dir, f'checkpoint_{epoch}.pkl')
    torch.save(state_dict, filename)
    # _run.add_artifact(filename)


@exp.automain
def run(device, loss_type, source, cuda, batch_size, checkpoint,
        epochs, latent_dim, model_type, max_iter, sigma, prob_param,
        gradient_reversal, regularization, lr, _seed, _run):
    torch.manual_seed(_seed)
    device = torch.device(f"cuda:{device}" if cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    if source == 'mnist':
        if loss_type == 'bce':
            transform = transforms.ToTensor()
        elif loss_type in ['kl', 'geometric', 'adversarial']:
            transform = transforms.Compose([transforms.ToTensor(), ToProb()])
        else:
            raise ValueError(f'Wrong `loss_type` argument, got {loss_type}')
        train_data = datasets.MNIST(expanduser('~/data/mnist'), train=True,
                                    download=True,
                                    transform=transform)
        test_data = datasets.MNIST(expanduser('~/data/mnist'), train=False,
                                   transform=transform)
        h, w = 28, 28
    elif 'quickdraw' in source:
        _, class_name, size = source.split('_')
        size = int(size)
        data_dir = join(expanduser('~/data/quickdraw/bitmaps'))
        h = w = size
        data = {}
        for fold in ['train', 'test']:
            filename = join(data_dir,
                            f'{class_name}_{size}_{fold}'
                            f'{"_norm" if loss_type != "bce" else ""}.pkl')
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

    gspace = Gspace2d(h, w, sigma=sigma, tol=1e-4, max_iter=max_iter,
                      method='lbfgs',
                      verbose=False)
    model = VAE(h, w, latent_dim,
                loss_type=loss_type, model_type=model_type,
                gspace=gspace, regularization=regularization,
                prob_param=prob_param,
                gradient_reversal=gradient_reversal)

    model = model.to(device)

    if loss_type == 'adversarial' and not gradient_reversal:
        optimizer = optim.Adam(list(model.decoder.parameters()) +
                               list(model.encoder.parameters()), lr=lr)
        reverse_optimizer = optim.Adam(model.prob_decoder.parameters())
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
        reverse_optimizer = None

    if checkpoint:
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict['model'])
        if reverse_optimizer is not None:
            reverse_optimizer.load_state_dict(state_dict['reverse_optimizer'])
        optimizer.load_state_dict(state_dict['reverse_optimizer'])
        start_epoch = state_dict['epoch']
    else:
        start_epoch = 0

    output_dir = join(base_dir, str(_run._id), 'artifacts')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for epoch in range(start_epoch, epochs + 1):
        train(model=model, optimizer=optimizer,
              reverse_optimizer=reverse_optimizer,
              loader=train_loader,
              epoch=epoch, device=device)
        _test(model=model, loader=test_loader, epoch=epoch, device=device,
              gspace=gspace,  # To track Hausdorff distance
              output_dir=output_dir)
        generate(model=model, epoch=epoch, device=device,
                 output_dir=output_dir)
        if epoch % 50 == 0:
            save_checkpoint(model=model, optimizer=optimizer,
                            reverse_optimizer=reverse_optimizer,
                            epoch=epoch,
                            output_dir=output_dir)
