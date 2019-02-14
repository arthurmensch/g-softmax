import os
from contextlib import nullcontext
from os.path import join, expanduser

import matplotlib.pyplot as plt
import numpy as np
import torch
from gsoftmax.continuous import MeasureDistance, DeepLoco
from gsoftmax.datasets import SyntheticSMLMDataset, SMLMDataset
from sacred import Experiment
from sacred import SETTINGS
from sacred.observers import FileStorageObserver
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

SETTINGS.HOST_INFO.INCLUDE_GPU_INFO = False

base_dir = expanduser('~/output/g-softmax/smlm')
exp = Experiment('vae')
exp.observers.append(FileStorageObserver.create(base_dir))


@exp.config
def system():
    device = 'cpu'
    seed = 100
    checkpoint = None
    log_interval = 100

    n_jobs = 4

    test_only = False


@exp.config
def base():
    test_source = 'MT0.N1.LD'
    modality = 'AS'
    dimension = 3

    loss = 'sinkhorn'
    coupled = False
    terms = 'symmetric'
    kernel = 'energy_squared'

    batch_size = 1024
    train_size = int(1024 * 1e3)
    eval_size = 2048
    test_size = None

    distance_type = 2

    sigmas = [0.3]
    epsilon = 1
    rho = 1e-1
    lr = 1e-3

    n_epochs = 100


@exp.named_config
def test_only():
    device = 'cpu'
    test_only = True
    checkpoint = None


class ModelLoss(nn.Module):
    def __init__(self, model, loss_fns):
        super().__init__()
        self.model = model
        self.loss_fns = nn.ModuleList(loss_fns)

    def forward(self, imgs, positions, weights):
        pred_positions, pred_weights = self.model(imgs)
        loss = 0
        for loss_fn in self.loss_fns:
            loss += loss_fn(pred_positions, pred_weights, positions, weights)
        return loss, pred_positions, pred_weights


def plot_example(datasets, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(7, 3))

    for ax, (fold, dataset) in zip(axes, datasets.items()):
        img, positions, weights = dataset[np.random.randint(0, len(dataset))]
        c, m, n = img.shape
        ax.imshow(img[0])
        ax.scatter(positions[:, 0] * m,
                   positions[:, 1] * n,
                   s=weights * 10, color='red')
        ax.set_xlim([0, m])
        ax.set_ylim([0, n])
        ax.set_title(f'{fold} example')
        ax.axis('off')
    plt.show()
    plt.savefig(join(output_dir, 'examples.png'))
    plt.close(fig)


def plot_pred_ground_truth(pred_positions, pred_weights,
                           positions, weights,
                           img, filename):
    fig, axes = plt.subplots(1, 2, figsize=(5, 5))
    c, m, n = img.shape

    pred_positions = pred_positions.detach().cpu()
    positions = positions.detach().cpu()
    weights = weights.detach().cpu()
    pred_weights = pred_weights.detach().cpu()
    img = img.detach().cpu()

    for ax, p, w in zip(axes, (pred_positions, positions),
                        (pred_weights, weights)):
        ax.imshow(img[0])
        ax.scatter(p[:, 0] * m,
                   p[:, 1] * n,
                   s=w * 10, color='red')
        ax.set_xlim([0, 64])
        ax.set_ylim([0, 64])
        ax.axis('off')
    plt.savefig(filename)
    plt.close(fig)


def metrics(pred_positions, pred_weights, positions, weights, offset, scale,
            reduction='mean', threshold=50):
    dim = pred_positions.shape[2]
    jaccards = []
    rmses_xy = []
    rmses_z = []

    pred_positions = pred_positions.detach().cpu()
    positions = positions.detach().cpu()
    weights = weights.detach().cpu()
    pred_weights = pred_weights.detach().cpu()

    pred_positions = pred_positions * scale
    positions = positions * scale
    pred_positions = pred_positions + offset
    positions = positions + offset

    for ppos, pweight, pos, weight in zip(pred_positions, pred_weights,
                                          positions, weights):
        ppos = ppos[pweight > 0]

        pos = pos[weight > 0]

        if len(pos) == 0 or len(ppos) == 0:
            jaccard = 0.
            rmse_xy = 0.
            rmse_z = 0.
        else:
            cost_matrix = pairwise_distances(ppos, pos, metric='euclidean')
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            match = cost_matrix[row_ind, col_ind]
            fn = max(0, len(pos) - len(ppos))
            tp = (match < threshold).sum()
            fp = len(match) - tp
            jaccard = float(tp) / (fn + fp + tp)

            if dim == 2:
                rmse_xy = match.sum()
                rmse_z = 0.
            else:
                cost_matrix_xy = pairwise_distances(ppos[:, :2], pos[:, :2],
                                                    metric='euclidean')
                cost_matrix_z = pairwise_distances(ppos[:, 2:3], pos[:, 2:3],
                                                   metric='euclidean')
                rmse_xy = np.sqrt((cost_matrix_xy[row_ind, col_ind] ** 2).mean())
                rmse_z = np.sqrt((cost_matrix_z[row_ind, col_ind] ** 2).mean())

        jaccards.append(jaccard)
        rmses_xy.append(rmse_xy)
        rmses_z.append(rmse_z)
    jaccard = torch.tensor(jaccards)
    rmse_xy = torch.tensor(rmses_xy)
    rmse_z = torch.tensor(rmse_z)
    if reduction == 'mean':
        jaccard = jaccard.mean()
        rmse_xy = rmse_xy.mean()
        rmse_z = rmse_z.mean()
    elif reduction == 'sum':
        jaccard = jaccard.sum()
        rmse_xy = rmse_xy.sum()
        rmse_z = rmse_z.sum()
    return jaccard, rmse_xy, rmse_z


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def save_checkpoint(model, optimizer, filename):
    state_dict = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict()}
    torch.save(state_dict, filename)


def load_checkpoint(filename, model, optimizer=None):
    state_dict = torch.load(filename)
    model.load_state_dict(state_dict['model'])
    if optimizer is not None:
        optimizer.load_state_dict(state_dict['optimizer'])


@exp.capture
def train_eval_loop(model_loss, loader, fold, epoch, output_dir,
                    device, log_interval, _run, optimizer=None, train=False,
                    ):
    records = dict(loss=0.)
    if not train:
        records['jaccard'] = 0.
        records['rmse_xy'] = 0.
        records['rmse_z'] = 0.
    n_samples = 0
    if train:
        model_loss.train()
    else:
        model_loss.eval()
        offset, scale, _, _, _ = loader.dataset.get_geometry()
    with torch.no_grad() if not train else nullcontext():
        for batch_idx, (imgs, positions, weights) in enumerate(loader):
            batch_size = imgs.shape[0]

            imgs = imgs.to(device)
            positions = positions.to(device)
            weights = weights.to(device)
            if train:
                optimizer.zero_grad()
            loss, pred_positions, pred_weights = model_loss(imgs, positions, weights)

            if train:
                loss.backward()
            else:
                jaccard, rmse_xy, rmse_z = metrics(
                    pred_positions, pred_weights, positions, weights,
                    offset=offset, scale=scale)
                records['jaccard'] += jaccard * batch_size
                records['rmse_xy'] += rmse_xy * batch_size
                records['rmse_z'] += rmse_z * batch_size

            if batch_idx == 0:
                for i in range(1):
                    plot_pred_ground_truth(pred_positions[i],
                                           pred_weights[i],
                                           positions[i],
                                           weights[i],
                                           imgs[i],
                                           join(output_dir,
                                                f'img_{fold}_e_{epoch}_{i}.png'))
            records['loss'] += loss.item() * batch_size
            n_samples += batch_size
            if train:
                optimizer.step()
                if batch_idx % log_interval == 0:
                    print(f'{fold} epoch: {epoch} '
                          f'[{batch_idx * batch_size}/{len(loader.dataset)} '
                          f'({100. * batch_idx / len(loader):.0f}%)]\t'
                          f'mean weights: {pred_weights.sum():.3f}\t'
                          f'loss: {loss.item():.6f}')
    print(f"=================> {fold} epoch: {epoch}", flush=False)
    for name, record in records.items():
        print(f'{fold}.{name} : {record / n_samples} ', flush=False)
        _run.log_scalar(f'{fold}.{name}', record / n_samples, epoch)
    print('')


@exp.main
def main(test_source, train_size, n_jobs,
         eval_size, batch_size, n_epochs, checkpoint,
         distance_type, test_only, dimension, test_size,
         loss, coupled, terms, kernel, sigmas, epsilon, rho, lr,
         modality, device, _seed, _run):
    output_dir = join(base_dir, str(_run._id), 'artifacts')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    torch.manual_seed(_seed)
    device = torch.device(f"cuda:{device}" if isinstance(device, int)
                          else "cpu")

    lengths = {'train': train_size, 'eval': eval_size}
    datasets = {}
    loaders = {}

    datasets['test'] = SMLMDataset(name=test_source, modality=modality,
                                   dimension=dimension,
                                   return_activation=False, length=test_size)
    loaders['test'] = DataLoader(datasets['test'], batch_size=batch_size,
                                 num_workers=1)
    # Use geometrical information to construct synthetics datasets
    offset, scale, shape, max_beads, w_range = datasets['test'].get_geometry()
    print(f'Dataset info: offset {offset}, scale {scale}, shape {shape},'
          f' max_beads {max_beads}, w_range {w_range}')
    for fold in ['train', 'eval']:
        datasets[fold] = SyntheticSMLMDataset(max_beads=max_beads, noise=100,
                                              length=lengths[fold],
                                              offset=offset,
                                              scale=scale,
                                              shape=shape,
                                              psf_radius=2000,
                                              w_range=w_range,
                                              batch_size=1,
                                              dimension=dimension,
                                              return_activation=False,
                                              modality=modality)
        loaders[fold] = DataLoader(datasets[fold],
                                   batch_size=batch_size,
                                   worker_init_fn=worker_init_fn,
                                   num_workers=n_jobs,
                                   pin_memory=torch.device.type == 'cuda',
                                   shuffle=fold == 'train')

    plot_example(datasets, output_dir=output_dir)

    model = DeepLoco(beads=50, dimension=dimension)
    loss_fns = []
    for sigma in sigmas:
        loss_fns.append(MeasureDistance(loss=loss,
                                        coupled=coupled,
                                        terms=terms,
                                        distance_type=distance_type,
                                        kernel=kernel,
                                        max_iter=100,
                                        sigma=sigma,
                                        graph_surgery='',
                                        verbose=False,
                                        epsilon=epsilon, rho=rho,
                                        reduction='mean'))
    loss_model = ModelLoss(model, loss_fns)

    if not test_only:
        optimizer = Adam(loss_model.parameters(), lr=lr, amsgrad=True)
    else:
        n_epochs = 1
        optimizer = None

    if checkpoint is not None:
        load_checkpoint(checkpoint, model, optimizer=optimizer)

    loss_model.to(device)

    for epoch in range(n_epochs):
        np.random.seed(0)
        if not test_only:
            train_eval_loop(loss_model, loaders['train'], 'train', epoch,
                            optimizer=optimizer,
                            train=True, output_dir=output_dir)
        train_eval_loop(loss_model, loaders['eval'], 'eval', epoch,
                        output_dir=output_dir)
        train_eval_loop(loss_model, loaders['test'], 'test', epoch,
                        output_dir=output_dir)
        if not test_only:
            save_checkpoint(model, optimizer,
                            join(output_dir, f'checkpoint_{epoch}.pkl'))


if __name__ == '__main__':
    exp.run_commandline()
