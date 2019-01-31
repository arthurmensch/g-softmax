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
from torch.nn.functional import nll_loss
from torch.utils.data import TensorDataset
from torchvision import datasets
from torchvision.transforms import transforms
from torchvision.utils import save_image

from gsoftmax.euclidean import _BaseGSpaceImpl
from gsoftmax.functional import safe_log

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
    source = 'quickdraw_plane_64'
    checkpoint = False
    log_interval = 100
    supervised_score = False


@exp.config
def base():
    batch_size = 512
    epochs = 100
    loss_type = 'bce'
    latent_dim = 256
    model_type = 'conv'
    max_iter = 30
    sigma = 1.
    epsilon = 1.
    regularization = 1
    epsilon = 10
    ramp = False
    model_type = 'conv'
    lr = 1e-3


class VAE(nn.Module):
    def __init__(self, h, w, latent_dim):
        super(VAE, self).__init__()
        self.h, self.w = h, w
        self.fc1 = nn.Linear(h * w, 400)
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, h * w)

    def encode(self, x):
        h1 = F.relu(self.fc1(x.view(-1, self.h * self.w)))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h = F.relu(self.fc3(z))
        z = self.fc4(h)
        return z.view(z.shape[0], 1, self.h, self.w)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)
        return logits, mu, logvar


class Interpolate(nn.Module):
    def __init__(self, scale_factor):
        super(Interpolate, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale_factor,
                                      mode='nearest',
                                      )
        return x


class ConvVAE(nn.Module):
    def __init__(self, h, w, latent_dim):
        super(ConvVAE, self).__init__()
        self.h, self.w = h, w
        nc = 1
        ndf = 64
        ngf = 64
        self.conv = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
        )
        self.fc31 = nn.Linear((ndf * 8) * 4 * 4, latent_dim, bias=False)
        self.fc32 = nn.Linear((ndf * 8) * 4 * 4, latent_dim, bias=False)
        self.fc3 = nn.Linear(latent_dim, 64 * (h - 11) * (w - 11))
        seq = [
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False)

        ]
        self.deconv = nn.Sequential(*seq)

    def encode(self, x):
        h = self.conv(x)
        h = h.view(h.shape[0], -1)
        return self.fc31(h), self.fc32(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        z = z[:, :, None, None]
        return self.deconv(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)
        return logits, mu, logvar


@exp.capture
def loss_function(logits, x, mu, logvar, loss_type, regularization,
                  epsilon, metric_softmax=None, ):
    logits = logits / epsilon
    n_batch, n_channel, h, w = logits.shape
    if loss_type == 'metric':
        x = x.view(-1, h, w)
        logits = logits.view(-1, h, w)
        mlse = metric_softmax.lse(logits)
        loss = mlse - torch.sum((logits * x).view(-1, h * w), dim=1)
        loss = loss.sum()
        # regularization /= (h * w)
    elif loss_type == 'kl':
        x = x.view(-1, h * w)
        logits = logits.view(-1, h * w)
        logits = F.log_softmax(logits, dim=1)
        loss = F.kl_div(logits, x, reduction='sum')
        # regularization /= (h * w)
    elif loss_type == 'bce':
        loss = F.binary_cross_entropy_with_logits(
            logits.view(n_batch, -1), x.view(n_batch, -1),
            reduction='sum')
    loss *= epsilon

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    penalty = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp())
    return loss + penalty * regularization, penalty


@exp.capture
def pred_function(logits, metric_softmax, epsilon, loss_type):
    n_batch, n_channel, h, w = logits.shape
    logits = logits / epsilon
    if loss_type == 'metric':
        pred = metric_softmax(logits.view(-1, h, w))
        pred = pred.view(n_batch, n_channel, h, w)
    elif loss_type == 'kl':
        pred = F.softmax(logits.view(-1, h * w),
                         dim=1).view(n_batch, n_channel, h, w)
    elif loss_type == 'bce':
        pred = torch.sigmoid(logits)
    return pred


@exp.capture
def train(model, optimizer, loader, metric_softmax, loss_type,
          device, log_interval, epoch, regularization, ramp, _run):
    model.train()
    train_loss = 0
    train_penalty = 0
    if ramp:
        this_reg = min(1., float(epoch) / 10) * regularization
    else:
        this_reg = regularization

    for batch_idx, (data, _) in enumerate(loader):
        data = data.to(device)
        optimizer.zero_grad()
        logits, mu, logvar = model(data)
        if batch_idx % log_interval == 0:
            logits.register_hook(lambda grad: print(
                f'dl/dlogits: {torch.mean(torch.abs(grad))}'))
        loss, penalty = loss_function(logits, data, mu, logvar,
                                      regularization=this_reg,
                                      loss_type=loss_type,
                                      metric_softmax=metric_softmax)
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
def _test(model, loader, metric_softmax, loss_type,
          device, epoch, output_dir, regularization, ramp,
          _run):
    model.eval()
    test_loss = 0.
    test_penalty = 0.
    kl_div = 0.
    bregman_div = 0.
    bce = 0.

    if ramp:
        this_reg = min(1., float(epoch) / 10) * regularization
    else:
        this_reg = regularization
    with torch.no_grad():
        for i, (data, _) in enumerate(loader):
            batch_size, n_channel, h, w = data.shape
            data = data.to(device)
            logits, mu, logvar = model(data)
            loss, penalty = loss_function(logits, data, mu, logvar,
                                          regularization=this_reg,
                                          loss_type=loss_type,
                                          metric_softmax=metric_softmax)
            test_loss += loss.item()
            test_penalty += penalty.item()
            pred = pred_function(logits, metric_softmax=metric_softmax,
                                 loss_type=loss_type,)

            kl_div += F.kl_div(safe_log(pred.view(batch_size, -1)),
                               target=data.view(batch_size, -1),
                               reduction='sum').item()

            metric_space = metric_softmax.metric_space
            max_iter = metric_space.max_iter
            tol = metric_space.tol
            metric_space.max_iter = 100
            metric_space.tol = 1e-5
            bregman_div += metric_space.hausdorff(pred.view(-1, h, w),
                                                  data.view(-1, h, w), reduction='sum').item()
            metric_space.max_iter = max_iter
            metric_space.tol = tol

            bce += F.binary_cross_entropy(pred.view(batch_size, -1),
                                          target=data.view(batch_size, -1),
                                          reduction='sum').item()

            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], pred[:n]])
                n_batch, n_channel = comparison.shape[:2]
                # Normalize
                comparison /= comparison.view(
                    n_batch, n_channel, -1).max(dim=2)[0][:, :, None, None]
                save_image(comparison.cpu(),
                           join(output_dir, 'reconstruction_' + str(epoch) +
                                '.png'), nrow=n)

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
def generate(model, metric_softmax, loss_type, latent_dim, device, output_dir, epoch):
    model.eval()
    with torch.no_grad():
        sample = torch.randn(64, latent_dim).to(device)
        logits = model.decode(sample)
        pred = pred_function(logits, metric_softmax, loss_type=loss_type,)
        # Normalize
        pred /= pred.view(
            pred.shape[0], pred.shape[1], -1).max(dim=2)[0][:, :, None, None]
        save_image(pred, join(output_dir, 'sample_' + str(epoch) + '.png', ))


@exp.capture
def save_checkpoint(model, optimizer, output_dir, epoch):
    state_dict = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'epoch': epoch}
    torch.save(state_dict, join(output_dir, f'checkpoint_{epoch}.pkl'))


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
        raise ValueError('Wrong `loss_type` argument'
                         '')

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

    metric_softmax = _BaseGSpaceImpl(h, w, epsilon=2, sigma=sigma, tol=1e-4,
                                     max_iter=max_iter, verbose=False).to(device)

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

    optimizer = optim.Adam(vae.parameters(), lr=lr)

    if checkpoint:
        state_dict = torch.load(checkpoint)
        vae.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        start_epoch = state_dict['epoch']
    else:
        start_epoch = 0

    output_dir = join(output_dir, str(exp._id), 'artifacts')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for epoch in range(start_epoch, epochs + 1):
        train(model=vae, optimizer=optimizer, loader=train_loader,
              epoch=epoch, metric_softmax=metric_softmax, device=device)
        _test(model=vae, loader=test_loader,
              epoch=epoch, metric_softmax=metric_softmax, device=device,
              output_dir=output_dir)
        generate(model=vae, epoch=epoch, metric_softmax=metric_softmax,
                 device=device, output_dir=output_dir)
        if epoch % 10 == 0:
            save_checkpoint(model=vae, optimizer=optimizer, epoch=epoch,
                            output_dir=output_dir)
