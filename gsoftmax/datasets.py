import glob
import os
import shutil
import urllib
import zipfile
from collections.abc import Iterable, Iterator
from os.path import expanduser, join

import joblib
import numpy as np
import pandas as pd
import scipy.interpolate
import torch
from PIL import Image
from sklearn.utils import check_random_state
from torch.utils.data import Dataset


def get_data_dir():
    return expanduser('~/data/deep_loco')


def fetch_smlm_dataset(name='MT0.N1.LD', modality='2D',
                       overwrite=False):
    assert name in ['MT0.N1.LD', 'MT0.N1.HD', 'Beads']

    base_url = f"http://bigwww.epfl.ch/smlm/challenge2016/datasets/{name}"
    if name == 'Beads':
        zip_url = f'{base_url}/Data/z-stack-{name}-{modality}-Exp-as-list.zip'
        activation_url = f'{base_url}/Data/activations.csv'
    else:
        zip_url = f'{base_url}/Data/sequence-{name}-{modality}-Exp-as-list.zip'
        activation_url = f'{base_url}/sample/activations.csv'
    dest_dir = join(get_data_dir(), name, modality)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    downloads = {activation_url: 'activation.csv',
                 zip_url: 'zstack.zip'}

    for url, filename in downloads.items():
        filename = join(dest_dir, filename)
        if not os.path.exists(filename) or overwrite:
            print(f'Trying to download {url}')
            with urllib.request.urlopen(url) as response, \
                    open(filename, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
            print(f'Done downloading {url}')
    zip_filename = join(dest_dir, downloads[zip_url])
    activation_filename = join(dest_dir, downloads[activation_url])
    dest_zip = join(dest_dir, 'zstack')
    if not os.path.exists(dest_zip) or overwrite:
        print(f'Extracting {zip_filename}')
        zip_ref = zipfile.ZipFile(zip_filename, 'r')
        zip_ref.extractall(join(dest_dir, 'zstack'))
        zip_ref.close()
    if name == 'Beads' and modality == 'DH-Exp':
        img_dir = 'zstack/sequence-as-stack-Beads-DH-Exp-as-list'
    else:
        img_dir = 'zstack'
    pkl_file = join(dest_dir, 'img.pkl')
    if not os.path.exists(pkl_file) or overwrite:
        imgs = sorted(glob.glob(join(dest_dir, img_dir, '*.tif')))
        img = np.r_[[np.array(Image.open(img)).astype(np.float32)
                     for img in imgs]]
        joblib.dump(img, pkl_file)
    activations = pd.read_csv(activation_filename, sep=',',
                              header=None if name == 'Beads' else 0,
                              dtype=np.float32,
                              names=['index', 'frame', 'x', 'y', 'z',
                                     'weight'])
    activations = activations.drop(columns='index')
    activations = activations.set_index('frame')
    activations.sort_index(inplace=True)

    positions, weights = zip(*((group.values[:, :3], group.values[:, 3])
                               for index, group in
                               activations.groupby(level='frame')))

    if name == 'Beads':
        affine = {'resx': 100, 'resy': 100, 'resz': 10,
                  'x0': 0, 'y0': 0, 'z0': -750}
    else:
        affine = {'resx': 100, 'resy': 100, 'resz': 10,
                  'x0': 0, 'y0': 0, 'z0': -750}

    return joblib.load(pkl_file, mmap_mode='r'), affine, positions, weights


class SMLMDataset(Dataset):
    def __init__(self, name='MT0.N1.LD', modality='2D', transform=None):
        self.imgs, self.affine, self.positions, self.weights = fetch_smlm_dataset(
            name=name, modality=modality)

        self.transform = transform

    def __getitem__(self, index):
        positions = torch.from_numpy(self.positions[index])
        weights = torch.from_numpy(self.weights[index])
        img = torch.from_numpy(self.imgs[index])

        if self.transform is not None:
            img = self.transform(img)

        return img[None, :], positions, weights

    def __len__(self):
        return len(self.imgs)


class ForwardModel(object):
    def __init__(self, psf_radius=2700.0, background_noise=100,
                 name='Beads', modality='2D',
                 random_state=None):
        """
        Use saved data from a z-stack to generate simulated STORM images
        """

        self.psf_radius = psf_radius
        self.background_noise = background_noise
        self.random_state = check_random_state(random_state)
        imgs, self.affine, positions, weights = fetch_smlm_dataset(name=name,
                                                                   modality=modality)
        positions = np.array(positions)
        weights = np.array(weights)
        self.offsets = positions[0, :, :2]
        self.weights = weights[0]

        imgs = np.array(imgs) - self.background_noise
        k, m, n = imgs.shape

        imgs[:, 0, :] = 0.0
        imgs[:, -1, :] = 0.0
        imgs[:, :, 0] = 0.0
        imgs[:, 0, -1] = 0.0
        self.splines = [scipy.interpolate.RectBivariateSpline(
            np.linspace(0, m * self.affine['resx'], m, endpoint=False),
            np.linspace(0, n * self.affine['resy'], n, endpoint=False),
            img, kx=1, ky=1) for img in imgs]

    def draw(self, img, x, y, z, w):
        """
        Render a point source at (x, y, z) nm with weight w
        onto the passed-in img.
        """
        m, n = img.shape

        aff = self.affine

        gridx = np.linspace(0.0, m * self.affine['resx'], m, endpoint=False)
        gridy = np.linspace(0.0, n * aff['resy'], n, endpoint=False)

        z_scaled = (z - self.affine['z0']) / self.affine['resz']
        zl = int(np.floor(z_scaled))
        zu = int(np.ceil(z_scaled))

        x_mask = abs(gridx - x) < self.psf_radius
        y_mask = abs(gridy - y) < self.psf_radius
        x_nnz = x_mask.nonzero()[0]
        y_nnz = y_mask.nonzero()[0]
        x_slice = slice(x_nnz.min(), x_nnz.max() + 1)
        y_slice = slice(y_nnz.min(), y_nnz.max() + 1)

        n_beads = len(self.offsets)
        bead = self.random_state.randint(0, n_beads)
        xs, ys = self.offsets[bead]
        weight = self.weights[bead]

        alpha = zu - z_scaled

        # For some reason we need this offset
        fx = (gridx + (xs - x))[x_slice]
        fy = (gridx + (ys - y))[y_slice]
        imgl = self.splines[zl](fy, fx)
        imgu = self.splines[zu](fy, fx)
        img[y_slice, x_slice] += ((alpha * imgl + (1 - alpha) * imgu)
                                  * w / weight)
        return img

    def sample(self, thetas, weights):
        """
        Generate a batch empirically.
        """
        batch_size = thetas.shape[0]
        length = thetas.shape[1]
        assert thetas.shape == (batch_size, length, 3)
        assert weights.shape == (batch_size, length)

        imgs = np.zeros((batch_size, 64, 64), dtype=thetas.dtype)

        for img, theta, weight in zip(imgs, thetas, weights):
            for ((x, y, z), w) in zip(theta, weight):
                if w != 0.0:
                    self.draw(img, x, y, z, w)
        return imgs


class EMCCD(object):
    """
    From SMLM Website.
    """

    def __init__(self, background_noise=0.0, quantum_efficiency=0.9,
                 read_noise=74.4, spurious_charge=0.0002, em_gain=300.0,
                 baseline=100.0, e_per_adu=45.0,
                 random_state=None):
        self.quantum_efficiency = quantum_efficiency
        self.read_noise = read_noise
        self.spurious_charge = spurious_charge
        self.em_gain = em_gain
        self.baseline = baseline
        self.e_per_adu = e_per_adu
        self.background_noise = background_noise
        self.random_state = check_random_state(random_state)

    def add_noise(self, photon_counts):
        n_ie = self.random_state.poisson(
            self.quantum_efficiency * (photon_counts + self.background_noise) + self.spurious_charge)
        n_oe = self.random_state.gamma(n_ie + 0.001, scale=self.em_gain)
        n_oe = n_oe + self.random_state.normal(0.0, self.read_noise,
                                               n_oe.shape)
        adu_out = (n_oe / self.e_per_adu).astype(int) + self.baseline
        return self.center(np.minimum(adu_out, 65535))

    def gain(self):
        return self.quantum_efficiency * self.em_gain / self.e_per_adu

    def mean(self):
        return (self.background_noise * self.quantum_efficiency + self.spurious_charge) * self.em_gain / \
               self.e_per_adu + self.baseline

    def center(self, img):
        return (img - self.mean()) / self.gain()


class UniformCardinalityPrior(object):
    def __init__(self, n=10, random_state=None):
        self.min, self.max, self.min_w, self.max_w = (
            np.array([0, 0, -750]), np.array([6400, 6400, 750]), 1000, 7000)
        self.n = n
        self.random_state = check_random_state(random_state)

    def sample(self, batch_size):
        weights = self.random_state.uniform(self.min_w, self.max_w,
                                            (batch_size, self.n))
        #  each frame gets a number of sources that is uniform in {0, ..., N}
        n_sources = self.random_state.randint(0, self.n + 1, batch_size)
        for b_idx in range(batch_size):
            weights[b_idx, :n_sources[b_idx]] = 0.0
        thetas = self.random_state.uniform(low=self.min, high=self.max,
                                           size=(
                                               batch_size, self.n,
                                               len(self.min)))
        return thetas, weights


class SyntheticSMLMDataset(Dataset):
    def __init__(self, n_beads=3, noise=100, batch_size=256,
                 length=100, random_state=None, ):
        self.parameter_prior = UniformCardinalityPrior(
            n_beads, random_state=random_state)
        self.forward_model = ForwardModel(background_noise=noise,
                                          random_state=random_state)
        self.affine = self.forward_model.affine
        self.noise_model = EMCCD(background_noise=noise,
                                 random_state=random_state)

        self.batch_size = batch_size
        self.length = length

    def __getitem__(self, i):
        images, positions, weights = self.sample()
        if self.batch_size == 1:
            return images[0][None, :], positions[0], weights[0]
        else:
            return images[:, None], positions, weights

    def __len__(self):
        return self.length

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        positions, weights = self.parameter_prior.sample(batch_size)
        positions = positions.astype('float32')
        weights = weights.astype('float32')
        noiseless = self.forward_model.sample(positions, weights)
        images = self.noise_model.add_noise(noiseless).astype('float32')
        return (torch.from_numpy(images), torch.from_numpy(positions),
                torch.from_numpy(weights))
