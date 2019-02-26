import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock


class ResLinear(nn.Module):
    def __init__(self, n_features, bias=True, batch_norm=False):
        super().__init__()
        self.l1 = nn.Linear(n_features, n_features, bias=bias)
        self.l2 = nn.Linear(n_features, n_features, bias=False)
        if batch_norm:
            self.bn1 = nn.BatchNorm1d(n_features)
            self.bn2 = nn.BatchNorm1d(n_features)
        self.batch_norm = batch_norm

    def forward(self, x):
        x = self.l1(x)
        if self.batch_norm:
            y = self.bn1(x)
        else:
            y = x
        y = self.l2(F.relu(y))
        if self.batch_norm:
            y = self.bn2(y)
        return y + x


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


class CNNPos(nn.Module):
    def __init__(self, architecture='deep_loco', zero=1e-12,
                 beads=10, dimension=3, batch_norm=False):
        super().__init__()

        if architecture == 'deep_loco':
            shapes = [[1, 16, 5, 2, 1],
                      [16, 16, 5, 2, 1],
                      [16, 64, 2, 0, 2],
                      [64, 64, 3, 1, 1],
                      [64, 256, 2, 0, 2],
                      [256, 256, 3, 1, 1],
                      [256, 256, 3, 1, 1],
                      [256, 256, 4, 0, 4]
                      ]
            seq = []
            for i, o, k, p, s in shapes:
                seq.append(nn.Conv2d(i, o, k, padding=p, stride=s))
                if batch_norm:
                    seq.append(nn.BatchNorm2d(o))
                seq.append(nn.ReLU(True))

            seq.append(Flatten(), )

            seq += [nn.Linear(4096, 2048),
                    ResLinear(2048, batch_norm=batch_norm),
                    ResLinear(2048, batch_norm=batch_norm)]

            self.barebone = nn.Sequential(*seq)
            state_size = 2048
        elif architecture == 'dcgan':
            nc = 1
            ndf = 64
            self.barebone = nn.Sequential(
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
                Flatten()
                # state size. (ndf*8) x 4 x 4
            )
            state_size = ndf * 8 * 4 * 4

        elif architecture == 'resnet':
            self.barebone = nn.Sequential(resnet18())
            state_size = 1000

        self.pos_fc = nn.Sequential(nn.Linear(state_size, beads * dimension, ))

        self.weight_fc = nn.Sequential(nn.Linear(state_size, beads))

        self.dimension = dimension
        self.beads = beads
        self.zero = zero
        self.batch_norm = batch_norm

    def forward(self, x):
        x = self.barebone(x)
        multiplier = 7 if self.batch_norm else 20
        offset = 10 / x.shape[1]
        position = torch.sigmoid(self.pos_fc(x).view(-1, self.beads,
                                                     self.dimension)
                                 * multiplier)
        weights = self.weight_fc(x) + offset
        weights = torch.clamp(weights, min=self.zero)
        return position, weights
