from torch.utils.data import DataLoader

from gsoftmax.datasets import SyntheticSMLMDataset, SMLMDataset


kernel_sigmas = [64.0, 320.0, 640.0, 1920.0]

n_beads = 5
lengths = {'train': 256000, 'eval': 1024}
datasets = {}
loaders = {}
for fold in ['train', 'eval']:
    datasets[fold] = SyntheticSMLMDataset(n_beads=n_beads, noise=100,
                                          length=lengths[fold],
                                          batch_size=1,
                                          random_state=None)
    loaders[fold] = DataLoader(datasets[fold],
                               batch_size=256, num_workers=4)

datasets['test'] = SMLMDataset(name='MT0.N1.LD', modality='2D')
loaders['test'] = DataLoader(datasets[fold], batch_size=256, num_workers=4)

import matplotlib.pyplot as plt

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
plt.show()
