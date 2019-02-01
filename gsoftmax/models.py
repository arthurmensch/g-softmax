import copy

import torch
from gsoftmax.modules import Gspace2d
from torch import nn
from torch.nn import functional as F


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        print('Reversing')
        return - grad_output


class GradientStop(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.fill_(0.)


class Encoder(nn.Module):
    def __init__(self, h, w, latent_dim):
        super().__init__()
        self.h, self.w = h, w
        self.fc1 = nn.Linear(h * w, 400)
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)

    def forward(self, x):
        h1 = F.relu(self.fc1(x.view(-1, self.h * self.w)))
        return self.fc21(h1), self.fc22(h1)


class Decoder(nn.Module):
    def __init__(self, h, w, latent_dim):
        super().__init__()
        self.h, self.w = h, w
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, h * w)

    def forward(self, z):
        h = F.relu(self.fc3(z))
        z = self.fc4(h)
        return z.view(z.shape[0], 1, self.h, self.w)


class ConvEncoder(nn.Module):
    def __init__(self, h, w, latent_dim):
        super().__init__()
        self.h, self.w = h, w
        nc = 1
        ndf = 64
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

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.shape[0], -1)
        return self.fc31(h), self.fc32(h)


class ConvDecoder(nn.Module):
    def __init__(self, h, w, latent_dim):
        super().__init__()
        self.h, self.w = h, w
        nc = 1
        ngf = 64
        self.deconv = nn.Sequential([
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
        ])

    def forward(self, z):
        z = z[:, :, None, None]
        return self.deconv(z)


class LastLayer(nn.Module):
    def __init__(self, loss_type='bce', gspace: Gspace2d = None):
        super().__init__()
        self.loss_type = loss_type

        if self.loss_type in ['geometric', 'adversarial']:
            self.gspace = gspace

    def pred(self, logits):
        if self.loss_type == 'adversarial':
            logits, prob = logits
        n_batch, n_channel, h, w = logits.shape

        if self.loss_type == 'adversarial':
            prob = prob.view(-1, h * w)

        logits = logits.view(-1, h * w)
        if self.loss_type == 'bce':
            rec = torch.sigmoid(logits)
        elif self.loss_type == 'kl':
            rec = F.softmax(logits, dim=1)
        elif self.loss_type == 'geometric':
            rec = self.gspace.softmax(logits)
        elif self.loss_type == 'adversarial':
            rec = self.gspace.softmax(prob)
        else:
            raise ValueError
        return rec.view(n_batch, n_channel, h, w)

    def forward(self, logits, target):
        n_batch, n_channel, h, w = target.shape
        if self.loss_type == 'adversarial':
            logits, prob = logits
            prob = prob.view(-1, h * w)

        target = target.view(-1, h * w)
        logits = logits.view(-1, h * w)

        if self.loss_type == 'bce':
            loss = F.binary_cross_entropy_with_logits(
                logits, target, reduction='sum')
        elif self.loss_type == 'kl':
            logits = F.log_softmax(logits, dim=1)
            loss = F.kl_div(logits, target, reduction='sum') * h * w
        elif self.loss_type == 'geometric':
            mlse = self.gspace.lse(logits)
            loss = mlse - torch.sum(logits * target, dim=1)
            loss = loss.sum() * h * w
        elif self.loss_type == 'adversarial':
            prob = F.log_softmax(prob, dim=1)
            loss = (self.gspace.conjugate_obj(prob, logits)
                    - torch.sum(logits * target, dim=1))
            loss = loss.sum()
        else:
            raise ValueError
        return loss


class VAE(nn.Module):
    def __init__(self, h, w, latent_dim=20, model_type='flat',
                 loss_type='bce', gspace=None, regularization=1):
        super().__init__()
        if model_type == 'flat':
            self.encoder = Encoder(h, w, latent_dim)
            self.decoder = Decoder(h, w, latent_dim)
        elif model_type == 'conv':
            self.encoder = ConvEncoder(h, w, latent_dim)
            self.decoder = Decoder(h, w, latent_dim)
        self.loss_type = loss_type
        self.last_layer = LastLayer(loss_type, gspace)
        if loss_type == 'adversarial':
            self.prob_decoder = copy.deepcopy(self.decoder)
        self.regularization = regularization

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def pred_latent_and_logits(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decoder(z)

        if self.loss_type != 'adversarial':
            return logits, mu, logvar
        else:
            prob = GradientReversal.apply(self.prob_decoder(z.detach()))
            return (logits, prob), mu, logvar

    def forward(self, x, return_penalty=False):
        logits, mu, logvar = self.pred_latent_and_logits(x)
        loss = self.last_layer(logits, x)

        penalty = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return loss + penalty * self.regularization, penalty

    def pred(self, x):
        return self.last_layer.pred(self.pred_latent_and_logits(x)[0])
