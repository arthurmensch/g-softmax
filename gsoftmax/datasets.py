import glob
import os
import shutil
import urllib
import zipfile
from collections import Iterable, Iterator
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

    if name == 'Beads':
        affine = {'resx': 100, 'resy': 100, 'resz': 10,
                  'x0': 0, 'y0': 0, 'z0': -750}
    else:
        affine = {'resx': 100, 'resy': 100, 'resz': 10,
                  'x0': 0, 'y0': 0, 'z0': -750}

    return joblib.load(pkl_file, mmap_mode='r'), affine, activations


class SMLMDataset(Dataset):
    def __init__(self, name, modality):
        self.X, _, self.y = fetch_smlm_dataset(name=name, modality=modality)
        self.y = self.y.reindex(['frame'])

    def __getitem__(self, index):
        this_y = self.y.loc[index].values

        return (torch.from_numpy(self.X[index]),
                torch.from_numpy(this_y[:, :3]),
                torch.from_numpy(this_y[:, 4]))

    def __len__(self):
        return len(self.X)


class ForwardModel(object):
    def __init__(self, random_state=None):
        """
        Use saved data from a z-stack to generate simulated STORM images
        """

        self.random_state = check_random_state(random_state)
        imgs, _, activations = fetch_smlm_dataset(name='Beads', modality='2D')
        activations = activations.values

        z_vals = np.linspace(-750, 750, 151)
        z_stack = [activations[activations[:, 3] == z] for z in
                   [x for x in z_vals]]
        imgs = imgs - 100

        splines = []
        offsets = []
        for f_idx in range(151):
            xs, ys = z_stack[f_idx][:, 1], z_stack[f_idx][:, 2]
            ### linear approx
            #### ZERO OUT BOUNDARY....
            img = imgs[f_idx]
            img[0, :] = 0.0
            img[-1, :] = 0.0
            img[:, 0] = 0.0
            img[0, -1] = 0.0
            spline = scipy.interpolate.RectBivariateSpline(
                [r * 100 for r in range(150)], [r * 100 for r in range(150)],
                img, kx=1, ky=1)
            ### The contest z-stack images are improperly normalized.
            splines.append(spline)
            xs = np.delete(xs, [1, 5])
            ys = np.delete(ys, [1, 5])
            offsets.append((xs, ys))
        self.splines = splines
        self.offsets = offsets

    def draw(self, x, y, z, image, w):
        """
        Render a point source at (x, y, z) nm with weight w
        onto the passed-in image.
        """

        new_pixel_locations = np.linspace(0.0, 6400, 64)
        z_scaled = (z + 750) / 1500 * 150
        z_low = int(np.floor(z_scaled))
        z_high = int(np.ceil(z_scaled))
        max_dist = 2700.0
        x_filter = abs(new_pixel_locations - x) < max_dist
        y_filter = abs(new_pixel_locations - y) < max_dist
        x_l, x_u = x_filter.nonzero()[0].min(), x_filter.nonzero()[0].max() + 1
        y_l, y_u = y_filter.nonzero()[0].min(), y_filter.nonzero()[0].max() + 1

        n_beads = len(self.offsets[z_low][0])
        p_idx = self.random_state.randint(0, n_beads)
        spline_l = self.splines[z_low]
        spline_h = self.splines[z_high]
        alpha = 1.0 - (z_scaled - z_low)
        beta = 1.0 - alpha
        alpha *= w / (30000 * 6.)
        beta *= w / (30000 * 6.)
        (xs, ys) = self.offsets[z_low]
        x_s, y_s = xs[p_idx], ys[p_idx]

        f_y = (new_pixel_locations + (y_s - y))[y_filter]
        f_x = (new_pixel_locations + (x_s - x))[x_filter]
        d_image = spline_l(f_y - 100, f_x - 100)
        image[y_l:y_u, x_l:x_u] += alpha * d_image

        (xs, ys) = self.offsets[z_high]
        x_s, y_s = xs[p_idx], ys[p_idx]

        f_y = (new_pixel_locations + (y_s - y))[y_filter]
        f_x = (new_pixel_locations + (x_s - x))[x_filter]
        d_image = spline_h(f_y - 100, f_x - 100)
        image[y_l:y_u, x_l:x_u] += beta * d_image
        return image

    def run_model(self, thetas, weights):
        """
        Generate a batch empirically.
        """

        # These are numpy arrays
        batch_size = thetas.shape[0]
        MAX_N = thetas.shape[1]
        assert thetas.shape == (batch_size, MAX_N, 3)
        assert weights.shape == (batch_size, MAX_N)

        images = np.zeros((batch_size, 64, 64))

        for b_idx in range(batch_size):
            for (w, (x, y, z)) in zip(weights[b_idx], thetas[b_idx]):
                if w != 0.0:
                    self.draw(x, y, z, images[b_idx], w)
        return images


class EMCCD(object):
    """
    From SMLM Website.
    """

    def __init__(self, noise_background=0.0, quantum_efficiency=0.9,
                 read_noise=74.4, spurious_charge=0.0002, em_gain=300.0,
                 baseline=100.0, e_per_adu=45.0,
                 random_state=None):
        self.qe = quantum_efficiency
        self.read_noise = read_noise
        self.c = spurious_charge
        self.em_gain = em_gain
        self.baseline = baseline
        self.e_per_adu = e_per_adu
        self.noise_bg = noise_background
        self.random_state = check_random_state(random_state)

    def add_noise(self, photon_counts):
        n_ie = self.random_state.poisson(
            self.qe * (photon_counts + self.noise_bg) + self.c)
        n_oe = self.random_state.gamma(n_ie + 0.001, scale=self.em_gain)
        n_oe = n_oe + self.random_state.normal(0.0, self.read_noise,
                                               n_oe.shape)
        ADU_out = (n_oe / self.e_per_adu).astype(int) + self.baseline
        return self.center(np.minimum(ADU_out, 65535))

    def gain(n):
        return n.qe * n.em_gain / n.e_per_adu

    def mean(n):
        return (n.noise_bg * n.qe + n.c) * n.em_gain / n.e_per_adu + n.baseline

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


class GenerativeModelIter(Iterator):
    def __init__(self, generative_model):
        self.generative_model = generative_model
        self.n_iter = 0
        self.max_iter = generative_model.max_iter

    def __next__(self):
        self.n_iter += 1
        if self.n_iter > self.max_iter:
            raise StopIteration
        return self.generative_model.sample()


class GenerativeModel(Iterable):
    def __init__(self, parameter_prior, forward_model, noise_model,
                 batch_size=256, max_iter=100):
        self.parameter_prior = parameter_prior
        self.forward_model = forward_model
        self.noise_model = noise_model

        self.batch_size = batch_size
        self.max_iter = max_iter

    def __iter__(self):
        return GenerativeModelIter(self)

    def sample(self):
        positions, weights = self.parameter_prior.sample(self.batch_size)
        noiseless = self.forward_model.run_model(positions, weights)
        positions = positions.astype('float32')
        weights = weights.astype('float32')
        images = self.noise_model.add_noise(noiseless).astype('float32')
        return (torch.from_numpy(images), torch.from_numpy(positions),
                torch.from_numpy(weights))


def make_generative_model(n_beads=3, noise=100, batch_size=8,
                          max_iter=100, random_state=None):
    return GenerativeModel(UniformCardinalityPrior(n_beads,
                                                   random_state=random_state),
                           ForwardModel(random_state=random_state),
                           EMCCD(noise_background=noise,
                                 random_state=random_state),
                           batch_size=batch_size,
                           max_iter=max_iter)
