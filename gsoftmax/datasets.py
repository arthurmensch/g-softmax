import glob
import os
import shutil
import urllib
import zipfile
from os.path import expanduser, join

import joblib
import numpy as np
import pandas as pd
import scipy.interpolate
import torch
from PIL import Image
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
    # Numbered from 1
    activations['frame'] = activations['frame'].astype(np.int64) - 1
    activations = activations.set_index('frame')
    activations.sort_index(inplace=True)

    max_beads = int(
        activations['x'].groupby(level='frame').aggregate('count').max())
    w_range = (activations['weight'].min(), activations['weight'].max())

    if name == 'Beads':
        scale = np.array([15000., 15000., 1500.], dtype=np.float32)
        offset = np.array([0., 0., -750.], dtype=np.float32)
    else:
        scale = np.array([6400., 6400., 700.], dtype=np.float32)
        offset = np.array([0., 0., -350.], dtype=np.float32)

    return (joblib.load(pkl_file, mmap_mode='r'), offset, scale,
            activations, max_beads, w_range)


class SMLMDataset(Dataset):
    def __init__(self, name='MT0.N1.LD', modality='2D', dimension=3,
                 transform=None, return_activation=True, length=None):
        (imgs, offset, scale, self.activations, self.max_beads, self.w_range)\
            = fetch_smlm_dataset(name=name, modality=modality)

        self.transform = transform
        self.return_activation = return_activation

        assert dimension in [2, 3]
        self.dimension = dimension

        self.imgs = torch.from_numpy(imgs)

        if length is not None:
            self.imgs = self.imgs[:length]
            self.activations = self.activations[:length]

        self.offset = torch.from_numpy(offset)[:self.dimension]
        self.scale = torch.from_numpy(scale)[:self.dimension]
        self.shape = imgs.shape[1:]

    def __getitem__(self, index):
        img = self.imgs[index][None, :]

        if self.transform is not None:
            img = self.transform(img)

        positions = torch.zeros((self.max_beads, self.dimension))
        weights = torch.zeros((self.max_beads,))
        try:
            activation = torch.from_numpy(self.activations.loc[index].values)
            if len(activation.shape) == 1:
                activation = activation[None, :]
        except KeyError:
            pass
        else:
            i = len(activation)
            positions[-i:] = activation[:, :self.dimension]
            positions[-i:] -= self.offset[None, :]
            positions[-i:] /= self.scale[None, :]
            if self.return_activation:
                weights[-i:] = activation[:, 3]
            else:
                weights[-i:] = 1
        return img, positions, weights

    def get_geometry(self):
        return self.offset.clone(), self.scale.clone(), self.shape, self.max_beads, self.w_range

    def __len__(self):
        return len(self.imgs)


class SyntheticSMLMDataset(Dataset):
    def __init__(self, scale, offset, shape, w_range,
                 max_beads=3, noise=100, batch_size=1,
                 length=100, modality='2D', return_activation=True,
                 psf_radius=2700, dimension=3):

        self.forward_model = ForwardModel(modality=modality,
                                          background_noise=noise,
                                          psf_radius=psf_radius,)

        assert dimension in [2, 3]
        self.dimension = dimension

        if dimension == 2:
            # Be random in selecting the z-stack index (add variance)
            offset = torch.cat((offset,
                               torch.tensor([self.forward_model.z_offset])))
            scale = torch.cat((scale,
                              torch.tensor([self.forward_model.z_scale])))

        self.parameter_prior = UniformCardinalityPrior(
            max_beads, min=offset, max=offset + scale,
            min_w=w_range[0], max_w=w_range[1])
        self.noise_model = EMCCD(background_noise=noise,)

        self.batch_size = batch_size
        self.length = length

        self.return_activation = return_activation

        assert dimension in [2, 3]
        self.dimension = dimension

        self.offset = offset[:self.dimension]
        self.scale = scale[:self.dimension]
        self.shape = shape
        self.max_beads = max_beads
        self.w_range = w_range

    def __getitem__(self, i):
        images, positions, weights = self.sample()

        if self.batch_size == 1:
            images, positions, weights = images[0], positions[0], weights[0]
        return images, positions, weights

    def __len__(self):
        return self.length

    def get_geometry(self):
        return self.offset.clone(), self.scale.clone(), self.shape,\
               self.max_beads, self.w_range

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        positions, weights = self.parameter_prior.sample(batch_size)
        positions = positions.astype('float32')
        weights = weights.astype('float32')
        noiseless = self.forward_model.sample(positions, weights,
                                              self.offset, self.scale,
                                              self.shape)
        images = self.noise_model.add_noise(noiseless).astype('float32')

        images = torch.from_numpy(images)
        positions = torch.from_numpy(positions)
        weights = torch.from_numpy(weights)

        images = images[:, None]

        if self.dimension == 2:
            positions = positions[:, :, :2]

        positions -= self.offset[None, :]
        positions /= self.scale[None, :]

        if not self.return_activation:
            weights[weights != 0] = 1

        return images, positions, weights


class ForwardModel(object):
    def __init__(self, modality='2D', psf_radius=2700.0,
                 background_noise=100, ):
        """
        Use saved data from a z-stack to generate simulated STORM images
        """

        zstack, source_offset, source_scale, activations, _, _ = \
            fetch_smlm_dataset(name='Beads', modality=modality)

        self.psf_radius = psf_radius
        self.background_noise = background_noise

        self.centers = activations.loc[0].values[:, :2]
        self.weights = activations.loc[0].values[:, 3]

        zstack = np.array(zstack) - self.background_noise
        k, m, n = zstack.shape

        zstack[:, 0, :] = 0.0
        zstack[:, -1, :] = 0.0
        zstack[:, :, 0] = 0.0
        zstack[:, 0, -1] = 0.0
        self.splines = [scipy.interpolate.RectBivariateSpline(
            np.linspace(source_offset[0], source_scale[0] + source_offset[0],
                        m, endpoint=False),
            np.linspace(source_offset[0], source_scale[1] + source_offset[1],
                        n, endpoint=False),
            img, kx=1, ky=1) for img in zstack]
        self.z_offset = source_offset[2]
        self.z_scale = source_scale[2]

    def draw(self, x, y, z, w, img, offset, scale):
        """
        Render a point source at (x, y, z) nm with weight w
        onto the passed-in img.
        """
        m, n = img.shape

        assert self.z_offset < z < self.z_offset + self.z_scale
        z_scaled = (z - self.z_offset) / self.z_scale
        zl = int(np.floor(z_scaled))
        zu = int(np.ceil(z_scaled))

        gridx = np.linspace(offset[0], scale[0] + offset[0], m, endpoint=False) - x
        gridy = np.linspace(offset[1], scale[1] + offset[1], n, endpoint=False) - y
        # gridx *= 3
        # gridy *= 3
        x_mask = abs(gridx) < self.psf_radius
        y_mask = abs(gridy) < self.psf_radius
        x_nnz = x_mask.nonzero()[0]
        y_nnz = y_mask.nonzero()[0]
        x_slice = slice(x_nnz.min(), x_nnz.max() + 1)
        y_slice = slice(y_nnz.min(), y_nnz.max() + 1)

        n_beads = len(self.centers)
        bead = np.random.randint(0, n_beads)
        xs, ys = self.centers[bead]
        weight = self.weights[bead]

        alpha = zu - z_scaled

        fx = (gridx + xs)[x_slice]
        fy = (gridy + ys)[y_slice]
        imgl = self.splines[zl](fy, fx)
        imgu = self.splines[zu](fy, fx)
        img[y_slice, x_slice] += ((alpha * imgl + (1 - alpha) * imgu)
                                  * w / weight)
        return img

    def sample(self, positions, weights, offset, scale, shape):
        """
        Generate a batch empirically.
        """
        batch_size = positions.shape[0]
        length = positions.shape[1]

        assert positions.shape == (batch_size, length, 3)
        assert weights.shape == (batch_size, length)

        imgs = np.zeros((batch_size, shape[0], shape[1]),
                        dtype=positions.dtype)

        for img, position, weight in zip(imgs, positions, weights):
            for ((x, y, z), w) in zip(position, weight):
                if w != 0.0:
                    self.draw(x, y, z, w, img, offset, scale)
        return imgs


class EMCCD(object):
    """
    From SMLM Website.
    """

    def __init__(self, background_noise=0.0, quantum_efficiency=0.9,
                 read_noise=74.4, spurious_charge=0.0002, em_gain=300.0,
                 baseline=100.0, e_per_adu=45.0,):
        self.quantum_efficiency = quantum_efficiency
        self.read_noise = read_noise
        self.spurious_charge = spurious_charge
        self.em_gain = em_gain
        self.baseline = baseline
        self.e_per_adu = e_per_adu
        self.background_noise = background_noise

    def add_noise(self, photon_counts):
        n_ie = np.random.poisson(
            self.quantum_efficiency * (
                    photon_counts + self.background_noise) + self.spurious_charge)
        n_oe = np.random.gamma(n_ie + 0.001, scale=self.em_gain)
        n_oe = n_oe + np.random.normal(0.0, self.read_noise,
                                       n_oe.shape)
        adu_out = (n_oe / self.e_per_adu).astype(int) + self.baseline
        return self.center(np.minimum(adu_out, 65535))

    def gain(self):
        return self.quantum_efficiency * self.em_gain / self.e_per_adu

    def mean(self):
        return (self.background_noise * self.quantum_efficiency
                + self.spurious_charge) * self.em_gain / \
               self.e_per_adu + self.baseline

    def center(self, img):
        return (img - self.mean()) / self.gain()


class UniformCardinalityPrior(object):
    def __init__(self, n, min, max, min_w, max_w):
        self.n = n
        self.min = min
        self.max = max
        self.min_w = min_w
        self.max_w = max_w

    def sample(self, batch_size):
        weights = np.random.uniform(self.min_w, self.max_w,
                                    (batch_size, self.n))
        #  each frame gets a number of sources that is uniform in {0, ..., N}
        # n_sources = np.random.poisson(self.n, batch_size)
        n_sources = np.random.randint(0, self.n + 1, batch_size)
        for b_idx in range(batch_size):
            weights[b_idx, :n_sources[b_idx]] = 0.0
        thetas = np.random.uniform(low=self.min, high=self.max,
                                   size=(batch_size, self.n,
                                         len(self.min)))
        return thetas, weights
