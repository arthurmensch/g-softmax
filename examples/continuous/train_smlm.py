import functools
import os
from contextlib import nullcontext
from os.path import join, expanduser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sacred import Experiment
from sacred import SETTINGS
from sacred.observers import FileStorageObserver
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from gsoftmax.continuous import CNNPos, Sum, EpsilonLR
from gsoftmax.datasets import SyntheticSMLMDataset, SMLMDataset
from gsoftmax.postprocessing import cluster_and_trim
from gsoftmax.sinkhorn import MeasureDistance

SETTINGS.HOST_INFO.INCLUDE_GPU_INFO = False

base_dir = expanduser('~/output/g-softmax/smlm')
exp = Experiment('vae')
exp.observers.append(FileStorageObserver.create(base_dir))


@exp.config
def system():
    device = 0
    seed = 200
    checkpoint = None
    log_interval = 100

    n_jobs = 8

    test_only = False
    train_only = False


@exp.config
def base():
    test_source = 'MT0.N1.LD'
    modality = 'AS'
    dimension = 3

    measure = 'sinkhorn'
    p = 2
    q = 2
    sigma = [1]
    epsilon = 1e-2
    rho = 1

    beads = 256

    zero = 1e-16

    architecture = 'resnet'
    batch_norm = False

    batch_size = 512
    train_size = int(1024 * 1024)
    eval_size = 2048
    test_size = 2048

    mass_norm = True

    lr = 1e-4
    gamma = 1

    n_epochs = 100

    repeat = False

    epsilon_gamma = 1.


@exp.named_config
def sinkhorn():
    measure = 'sinkhorn'
    sigma = [1]
    epsilon = 1e-2
    epsilon_gamma = 1

    batch_size = 512

    lr = 1e-4
    epoch = 100

    architecture = 'deep_loco'
    batch_norm = False
    mass_norm = True

@exp.named_config
def right_hausdorff():
    measure = 'right_hausdorff'
    sigma = [1]
    epsilon = 1e-1
    mass_norm = True
    lr = 1e-4
    epsilon_gamma = .5
    rho = 1


@exp.named_config
def mmd():
    measure = 'mmd'
    p = 1
    q = 1
    sigma = [0.01, 0.02, 0.04, 0.12]
    gamma = .5
    zero = 0
    batch_size = 1024
    train_size = 1024 * 1024
    n_epochs = 100
    lr = 1e-4
    epoch = 10

    architecture = 'deep_loco'
    batch_norm = False
    mass_norm = False


@exp.named_config
def single_batch():
    device = 'cpu'
    batch_size = 1
    train_size = int(1)
    eval_size = 1
    test_size = 10
    seed = 1000
    repeat = True
    train_only = False


@exp.named_config
def test_only():
    device = 'cpu'
    test_only = True
    checkpoint = None


class ClampedModelLoss(nn.Module):
    def __init__(self, model, loss_fn, zero=1e-7):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.zero = zero

    def forward(self, imgs, positions, weights):
        pred_positions, pred_weights = self.model(imgs)
        weights = torch.clamp(weights, min=self.zero)
        pred_weights = torch.clamp(pred_weights, min=self.zero)
        return self.loss_fn(pred_positions, pred_weights,
                            positions, weights), pred_positions, pred_weights


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
    plt.savefig(join(output_dir, 'examples.png'))
    plt.close(fig)


@exp.capture
def plot_pred_ground_truth(pred_positions, pred_weights,
                           positions, weights,
                           img, filename):
    fig, axes = plt.subplots(1, 2, figsize=(5, 5))
    c, m, n = img.shape

    for ax, pos, w in zip(axes, (pred_positions, positions),
                          (pred_weights, weights)):
        pos = pos[w > 0]
        w = w[w > 0]
        ax.imshow(img[0])
        ax.scatter(pos[:, 0] * m, pos[:, 1] * n,
                   s=w * 10, color='red')
        ax.set_xlim([0, 64])
        ax.set_ylim([0, 64])
        ax.axis('off')

    plt.savefig(filename)
    plt.close(fig)


@exp.capture
def metrics(pred_positions, pred_weights, positions, weights,
            reduction='mean', threshold=100):
    dim = pred_positions.shape[2]

    jaccards = []
    rmses_xy = []
    rmses_z = []

    for ppos, pweight, pos, weight in zip(pred_positions, pred_weights,
                                          positions, weights):
        ppos = ppos[pweight > 0]

        pos = pos[weight > 0]

        if len(pos) == 0 or len(ppos) == 0:
            rmse_xy = 0.
            rmse_z = 0.
            if len(pos) == 0 and len(ppos) > 0:
                tp = 0
                fp = len(ppos)
                fn = 0
                jaccard = 0.
            elif len(ppos) == 0 and len(pos) > 0:
                fn = len(ppos)
                tp = 0
                fp = 0
                jaccard = 0.
            else:
                fp = fn = tp = 0
                jaccard = 1.
        else:
            cost_matrix = pairwise_distances(pos, ppos, metric='euclidean')
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            matched_cost = cost_matrix[row_ind, col_ind]
            mask = matched_cost < threshold
            tp = mask.sum()
            fn = len(mask) - tp
            non_matched_mask = np.ones(cost_matrix.shape[1], dtype=np.bool)
            non_matched_mask[col_ind] = 0
            non_matched_cost = cost_matrix[:, non_matched_mask]
            fp = np.all(non_matched_cost > threshold, axis=0).sum()
            jaccard = float(tp) / (fn + fp + tp)

            if dim == 2:
                if tp > 0:
                    rmse_xy = np.sqrt((matched_cost[mask] ** 2).mean())
                else:
                    rmse_xy = 0.
                rmse_z = 0.
            else:
                cost_matrix_xy = pairwise_distances(pos[:, :2],
                                                    ppos[:, :2],
                                                    metric='euclidean')
                cost_matrix_z = pairwise_distances(pos[:, 2:3],
                                                   ppos[:, 2:3],
                                                   metric='euclidean')
                if tp > 0:
                    rmse_xy = np.sqrt(
                        (cost_matrix_xy[row_ind, col_ind][mask]
                         ** 2).mean())
                    rmse_z = np.sqrt((cost_matrix_z[row_ind, col_ind][mask]
                                      ** 2).mean())
                else:
                    rmse_xy = 0.
                    rmse_z = 0.
        jaccards.append(jaccard)
        # print(tp, fn, fp)
        rmses_xy.append(rmse_xy)
        rmses_z.append(rmse_z)

    jaccard = torch.tensor(jaccards)
    rmse_xy = torch.tensor(rmses_xy)
    rmse_z = torch.tensor(rmses_z)
    if reduction == 'mean':
        jaccard = jaccard.mean()
        rmse_xy = rmse_xy.mean()
        rmse_z = rmse_z.mean()
    elif reduction == 'sum':
        jaccard = jaccard.sum()
        rmse_xy = rmse_xy.sum()
        rmse_z = rmse_z.sum()
    return jaccard, rmse_xy, rmse_z


def worker_init_fn(worker_id, offset):
    np.random.seed((torch.initial_seed() + worker_id + offset) % (2 ** 31))


def save_checkpoint(model, optimizer, scheduler, filename):
    state_dict = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'scheduler': scheduler.state_dict()}
    torch.save(state_dict, filename)


def load_checkpoint(filename, model, optimizer=None, scheduler=None):
    state_dict = torch.load(filename)
    model.load_state_dict(state_dict['model'])
    if optimizer is not None:
        optimizer.load_state_dict(state_dict['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(state_dict['scheduler'])


@exp.capture
def train_eval_loop(model_loss, loader, fold, epoch, output_dir, dimension,
                    device, log_interval, _run, optimizer=None,
                    ):
    records = dict(loss=0.)
    records['jaccard'] = 0.
    records['rmse_xy'] = 0.
    records['rmse_z'] = 0.
    n_samples = 0
    if fold == 'train':
        model_loss.train()
    else:
        model_loss.eval()
    offset, scale, _, _, _ = loader.dataset.get_geometry()
    offset = offset.to(device)
    scale = scale.to(device)
    if fold == 'test':
        all_preds = []
    with torch.no_grad() if not fold == 'train' else nullcontext():
        for batch_idx, (imgs, positions, weights) in enumerate(loader):
            batch_size = imgs.shape[0]

            imgs = imgs.to(device)
            positions = positions.to(device)
            weights = weights.to(device)
            if fold == 'train':
                optimizer.zero_grad()
            loss, pred_positions, pred_weights = model_loss(imgs, positions,
                                                            weights)
            if fold == 'train':
                loss.backward()

            if fold != 'train' or batch_idx == 0:
                pred_positions = pred_positions * scale
                positions = positions * scale
                pred_positions = pred_positions + offset
                positions = positions + offset
                # Postprocess
                pred_positions, pred_weights = cluster_and_trim(pred_positions,
                                                                pred_weights,
                                                                0, 100,
                                                                1e-1)
                pred_positions = pred_positions.detach().cpu().numpy()
                pred_weights = pred_weights.detach().cpu().numpy()
                positions = positions.detach().cpu().numpy()
                weights = weights.detach().cpu().numpy()

                if fold != 'train' or batch_idx == 0:
                    size = len(loader.dataset) if fold == 'train' else batch_size
                    jaccard, rmse_xy, rmse_z = metrics(
                        pred_positions, pred_weights, positions, weights)
                    records['jaccard'] += jaccard * size
                    records['rmse_xy'] += rmse_xy * size
                    records['rmse_z'] += rmse_z * size

                if fold == 'test':
                    frames = np.repeat(np.arange(n_samples,
                                                 batch_size + n_samples,
                                                 dtype=np.float32)[:, None, None],
                                       pred_positions.shape[1], axis=1)
                    preds = np.concatenate([frames, pred_positions], axis=2)
                    preds = preds[pred_weights != 0]
                    all_preds.append(preds)

                if batch_idx == 0:
                    offset_ = offset.cpu().numpy()
                    scale_ = scale.cpu().numpy()
                    plot_pred_ground_truth((pred_positions[0] - offset_)
                                           / scale_, pred_weights[0],
                                           (positions[0] - offset_) / scale_,
                                           weights[0],
                                           imgs[0].detach().cpu(),
                                           join(output_dir,
                                                f'img_{fold}_e_{epoch}.png'))

            records['loss'] += loss.item() * batch_size
            n_samples += batch_size
            if fold == 'train':
                optimizer.step()
                if batch_idx % log_interval == 0:
                    print(f'{fold} epoch: {epoch} '
                          f'[{batch_idx * batch_size}/{len(loader.dataset)} '
                          f'({100. * batch_idx / len(loader):.0f}%)]\t'
                          f'loss: {loss.item():.6f}')
    if fold == 'test':
        all_preds = np.concatenate(all_preds, axis=0)
        if all_preds.shape[1] == 4:
            columns = ['frame', 'xnano', 'ynano', 'znano']
        else:
            columns = ['frame', 'xnano', 'ynano']
        df = pd.DataFrame(data=all_preds, columns=columns)
        df['frame'] = df['frame'].astype('int32')
        df['intensity'] = 1
        df.to_csv(join(output_dir, 'pred_activations.csv'))
    print(f"=================> {fold} epoch: {epoch}", flush=False)
    for name, record in records.items():
        print(f'{fold}.{name} : {record / n_samples} ', flush=False)
        _run.log_scalar(f'{fold}.{name}', record / n_samples, epoch)
    print('')


@exp.main
def main(test_source, train_size, n_jobs,
         eval_size, batch_size, n_epochs, checkpoint,
         test_only, dimension, test_size, architecture,
         measure, sigma, epsilon, rho, lr, zero, p, q, gamma,
         epsilon_gamma,
         batch_norm, train_only, beads, repeat,
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
                                              psf_radius=2700,
                                              w_range=w_range,
                                              batch_size=1,
                                              dimension=dimension,
                                              return_activation=False,
                                              modality=modality)
        loaders[fold] = DataLoader(datasets[fold],
                                   batch_size=batch_size,
                                   worker_init_fn=functools.partial(
                                       worker_init_fn,
                                       offset=1000 * (fold == 'eval')),
                                   num_workers=n_jobs,
                                   pin_memory=torch.device.type == 'cuda',
                                   shuffle=False)

    model = CNNPos(beads=beads, dimension=dimension, batch_norm=batch_norm,
                   architecture=architecture, zero=zero)

    if len(sigma) > 1:
        loss_fn = Sum(MeasureDistance(measure=measure, p=p, q=q,
                                      max_iter=100,
                                      sigma=this_sigma,
                                      epsilon=epsilon, rho=rho,
                                      reduction='mean')
                      for this_sigma in sigma)
        epsilon_scheduler = None
        # Desactivated for now
    else:
        if hasattr(sigma, '__iter__'):
            sigma = sigma[0]
        loss_fn = MeasureDistance(measure=measure, p=p, q=q,
                                  max_iter=100,
                                  sigma=sigma,
                                  epsilon=epsilon, rho=rho,
                                  reduction='mean')
        epsilon_scheduler = EpsilonLR(loss_fn, step_size=1,
                                      gamma=epsilon_gamma, min_epsilon=5e-3)
    loss_model = ClampedModelLoss(model, loss_fn, zero=zero)

    if not test_only:
        optimizer = Adam(loss_model.parameters(), lr=lr, amsgrad=True)
        scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    else:
        n_epochs = 1
        optimizer = None
        scheduler = None

    if checkpoint is not None:
        load_checkpoint(checkpoint, model, optimizer=optimizer,
                        scheduler=scheduler)

    loss_model.to(device)

    for epoch in range(n_epochs):
        if repeat:
            torch.manual_seed(_seed)
        else:
            torch.manual_seed(_seed + epoch)
        if not train_only:
            train_eval_loop(loss_model, loaders['eval'], 'eval', epoch,
                            output_dir=output_dir)
            train_eval_loop(loss_model, loaders['test'], 'test', epoch,
                            output_dir=output_dir)
        if not test_only:
            scheduler.step(epoch)
            if epsilon_scheduler is not None:
                epsilon_scheduler.step(epoch)
            train_eval_loop(loss_model, loaders['train'], 'train', epoch,
                            optimizer=optimizer,
                            output_dir=output_dir)
            if epoch % 10 == 0:
                save_checkpoint(model, optimizer, scheduler,
                                join(output_dir, f'checkpoint.pkl'))


if __name__ == '__main__':
    exp.run_commandline()
