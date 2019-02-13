from os.path import join, expanduser

import torch
from sacred import Experiment
from sacred.observers import FileStorageObserver
from torch.optim import Adam
from torch.utils.data import DataLoader

from gsoftmax.continuous import DeepLoco, MeasureDistance
from gsoftmax.datasets import SyntheticSMLMDataset, SMLMDataset

import matplotlib.pyplot as plt

import os

base_dir = expanduser('~/output/g-softmax/smlm')
exp = Experiment('vae')
exp.observers.append(FileStorageObserver.create(base_dir))


@exp.config
def system():
    cuda = True
    device = 0
    seed = 0
    checkpoint = False
    log_interval = 10


@exp.config
def base():
    test_source = 'MT0.N1.LD'
    modality = '2D'

    n_beads = 5

    loss = 'sinkhorn'
    coupled = False
    terms = 'left'
    sigmas = [1e-2]
    epsilon = 1
    rho = 1
    lr = 1e-3

    n_epochs = 10


def plot_example(output_dir, datasets):
    fig, axes = plt.subplots(1, 3, figsize=(7, 3))

    for ax, (fold, dataset) in zip(axes, datasets.items()):
        img, positions, weights = dataset[0]
        ax.imshow(img)
        ax.scatter(positions[:, 0] / dataset.affine['resx'],
                   positions[:, 1] / dataset.affine['resy'],
                   s=weights / 1000, color='red')
        ax.set_xlim([0, 64])
        ax.set_ylim([0, 64])
        ax.set_title(f'Fold example')
        ax.axis('off')
    plt.savefig(join(output_dir, 'examples.png'))


@exp.automain
def main(test_source, train_size,
         eval_size, batch_size,
         loss, coupled, terms, kernel, sigmas, epsilon, rho, lr,
         modality, device, cuda, n_beads, _seed, _run):
    output_dir = join(base_dir, str(_run._id), 'artifacts')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    torch.manual_seed(_seed)
    device = torch.device(f"cuda:{device}" if cuda else "cpu")

    lengths = {'train': train_size, 'eval': eval_size}
    datasets = {}
    loaders = {}
    for fold in ['train', 'eval']:
        datasets[fold] = SyntheticSMLMDataset(n_beads=n_beads, noise=100,
                                              length=lengths[fold],
                                              batch_size=batch_size,
                                              modality=modality,
                                              random_state=None)
        loaders[fold] = DataLoader(datasets[fold],
                                   batch_size=batch_size, num_workers=4)

    datasets['test'] = SMLMDataset(name=test_source, modality=modality)
    loaders['test'] = DataLoader(datasets[fold], batch_size=256, num_workers=1)

    model = DeepLoco()
    distances = []
    for sigma in sigmas:
        distances.append(MeasureDistance(loss=loss,
                                         coupled=coupled,
                                         terms=terms,
                                         distance_type=2,
                                         kernel=kernel,
                                         max_iter=100,
                                         sigma=sigma,
                                         graph_surgery='loop',
                                         verbose=False,
                                         epsilon=epsilon, rho=rho))
    optimizer = Adam(model.parameters(), lr=lr)

    history = {}
    for epoch in range(n_epochs):
        history[''] = 0
        seen_samples = 0
        for imgs, positions, weights in loaders['train']:
            batch_size = imgs.shape[0]
            imgs = imgs.to(device)
            positions = positions.to(device)
            weight = weight.to(device)

            optimizer.zero_grad()
            # Clamping prevent log barrier problems
            weights = torch.log(weights).clamp(min=-200)
            pred_positions, pred_weights = model(imgs)
            pred_weights = pred_weights.clamp_(min=-200)
            loss = 0.
            for distance in distances:
                loss += distance(pred_positions, pred_weights, positions,
                                 weights).mean()
            loss.backward()
            train_loss += loss.item() * batch_size
            seen_samples += batch_size
            optimizer.step()
        train_loss /= seen_samples

        for fold in ['eval', 'test']:

