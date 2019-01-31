import torch
from torch import nn
from torch.nn import functional as F


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


class LastLayer(nn.Module):
    def __init__(self, loss_type='bce', gspace=None):
        super().__init__()
        self.loss_type = loss_type

        if self.loss_type == 'geometric':
            self.gspace = gspace

    def forward(self, logits):
        n_batch, n_channel, h, w = logits.shape
        logits = logits.view(n_batch, -1)
        if self.loss_type == 'bce':
            rec = F.sigmoid(logits, dim=1)
        elif self.loss_type == 'kl':
            rec = F.softmax(logits, dim=1)
        elif self.loss_type == 'geometric':
            rec = self.gspace.softmax(logits, dim=1)
        else:
            raise ValueError
        return rec.view(n_batch, n_channel, h, w)

    def loss(self, logits, target):
        n_batch, n_channel, h, w = logits.shape

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
        else:
            raise ValueError
        return loss


class WrappedVAE(nn.Module):
    def __init__(self, model, last_layer, regularization=1):
        super().__init__()
        self.model = model
        self.last_layer = last_layer
        self.regularization = regularization

    def forward(self, x, return_penalty=False):
        logits, mu, logvar = self.model(x)
        loss = self.last_layer(logits, x)
        penalty = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return loss + penalty * self.regularization, penalty

    def decode(self, z):
        return self.last_layer(self.model.decode(z))

    def pred(self, x):
        logits, _, _ = self.model(x)
        return self.last_layer(logits)
