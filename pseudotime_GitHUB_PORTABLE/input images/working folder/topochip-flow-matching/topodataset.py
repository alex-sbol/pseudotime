import warnings

import torch
import torch.nn as nn
from torch import Tensor
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torch.utils.data.distributed import DistributedSampler
import os
from scipy.io import loadmat
from typing import Tuple
import matplotlib.pyplot as plt
import utils
from utils import plot_topo_cells
import numpy as np
from skimage.filters import threshold_otsu, gaussian, sobel
from skimage.morphology import remove_small_holes
from skimage.segmentation import watershed
from tqdm import tqdm
import pickle


class TopoDataSet(Dataset):

    def __init__(self, dir, downsampling_factor=1, limit_num_data=None, crop_size=None, threshold_input=False,
                 return_filename=False, ignore_last_10_p=False, get_only_last_10p=False):
        super().__init__()

        self.dir = dir
        self.files = []
        self.downsampling_factor = downsampling_factor
        self.crop_size = crop_size
        self.threshold_input = threshold_input
        # stats_per_chip = {}
        self.maxvals_per_ch = np.zeros(4)
        self.minvals_per_ch = np.zeros(4)
        for currentpath, folders, files in os.walk(dir):
            # if 'QualityCheckPassed' in currentpath:
            #     stats_per_chip[currentpath] = {'min': np.zeros(4)+np.inf, 'max': np.zeros(4)-np.inf, 'sum': np.zeros(4),
            #                                    'count': np.zeros(4)}
            files = sorted(files)
            assert not (get_only_last_10p and ignore_last_10_p), 'cannot get only last 10% of files and ignore last 10% at the same time!'
            if ignore_last_10_p:
                # ignore the last 10% of the files in each folder for validation purposes
                files = files[:int(len(files) * 0.9)]
            if get_only_last_10p:
                # get only the last 10% of the files in each folder for validation purposes
                files = files[int(len(files) * 0.9):]
            for file in tqdm(files, total=len(files)):
                if file.endswith('.mat'):
                    self.files.append(os.path.join(currentpath, file))
                    # im = loadmat(os.path.join(currentpath, file))['I'].astype('float')
                    # min_ch = np.nanmin(np.where(im>0, im, np.nan), axis=(0, 1))  # ignore zeros
                    # max_ch = np.max(im, axis=(0, 1))
                    # self.minvals_per_ch = np.minimum(self.minvals_per_ch, min_ch)
                    # self.maxvals_per_ch = np.maximum(self.maxvals_per_ch, max_ch)
                    # stats_per_chip[currentpath]['min'] = np.minimum(stats_per_chip[currentpath]['min'], min_ch)
                    # stats_per_chip[currentpath]['max'] = np.maximum(stats_per_chip[currentpath]['max'], max_ch)
                    # stats_per_chip[currentpath]['sum'] += np.sum(im, axis=(0, 1))
                    # stats_per_chip[currentpath]['count'] += np.ones(4) * im.shape[0] * im.shape[1]
                    # all_lastdims[currentpath][im.shape[0]] = 0 if im.shape[0] not in all_lastdims[currentpath].keys() else all_lastdims[currentpath][im.shape[0]] + 1

        # limit num data points when applicable
        all_file_len = len(self.files)
        self.files = self.files[:min(limit_num_data, len(self.files))] if limit_num_data is not None else self.files
        self.return_filename = return_filename

        with open('normalization_constants.pickle', 'rb') as handle:
            self.stats_per_chip = pickle.load(handle)

        # make sure self.files has approximately the same length as the original dataset
        while len(self.files) < all_file_len:
            self.files += self.files

        print('dataset contains {} images'.format(all_file_len))



    def __getitem__(self, idx, make_plot=False):
        fname = self.files[idx]
        mat = loadmat(os.path.join(fname))
        arr = mat['I'].astype(float)
        # arr /= 65535.
        # for some reason we multiplied by this in the matlab script? ask nikita
        # warnings.warn('for some reason we multiplied by this in the matlab script? ask nikita!')
        # arr[..., 0] = arr[..., 0]
        arr = arr[..., (0,3,2,1)]  # RED: YAP, GREEN: ACTIN, BLUE: DAPI
        # arr = utils.mask_approx_yap(arr)
        # crop the image from the center:
        if self.crop_size is not None:
            h, w = arr.shape[:2]
            crop = self.crop_size
            arr = arr[h//2-crop//2:h//2+crop//2, w//2-crop//2:w//2+crop//2]
        # arr = utils.normalize_channelwise(arr, do_simple_normalization=False)

        # normalize the array following the min/max exposures for each channel, for each insert:
        stats_this_file = self.stats_per_chip[os.path.dirname(fname)[5:16]]
        arr = utils.normalize_channelwise_per_chip(arr,
                                                   stats_this_file['min'][(0,3,2,1),],  # permute in the same way as the image channels
                                                   stats_this_file['max'][(0,3,2,1),]
                                                   )

        tensor = torch.from_numpy(arr)
        tensor = tensor.permute(2,0,1)
        if self.downsampling_factor > 1:
            tensor = nn.Upsample(scale_factor=1/self.downsampling_factor, mode='bilinear')(tensor.unsqueeze(0))
            tensor = tensor.squeeze(0)
            # print('downsampling factor is {}'.format(self.downsampling_factor))
        img = tensor[1:].float()
        bg = tensor[0:1].float()

        if self.threshold_input:
            # threshold the input image, keep only the activations within the mask. this needs to be done channel-wise
            actin_blurred = gaussian(img[1].numpy(), sigma=2)
            actin_mask = img[1] >= threshold_otsu(actin_blurred)
            actin_mask = remove_small_holes(actin_mask.numpy(), area_threshold=2500)
            actin_mask = torch.from_numpy(actin_mask.astype(float))
            img[-1] = img[-1] *  actin_mask +( -1) * (1 - actin_mask)  # dapi
            img[0] = img[0] * actin_mask + (-1) * (1 - actin_mask)   # yap
            dapi_within_actin = img[-1][actin_mask.bool().numpy()][None, ...].numpy()
            dapi_mask = (img[-1] >= threshold_otsu(dapi_within_actin)).float()
            img[-1] = img[-1] * dapi_mask.float() + (-1) * (1 - dapi_mask.float())
            # for c in range(img.shape[0]):
            #     mask = img[c] >= threshold_otsu(img[c].numpy())
            #     img[c] = img[c] * mask + (-1) * (1 - mask.long())

        out = (img, bg)  # separate conditioning from cell image. first img, then cond

        if self.return_filename:
            out = (*out, fname)
        return  out


    def __len__(self):
        return len(self.files)


if __name__ == '__main__':
    d = TopoDataSet('data/')
    np.random.seed(0)
    # indices = np.random.randint(0, len(d), 10)
    indices = [0, 1, 2, 3, 4, 5]
    # indices = np.nonzero(np.array(d.files) == 'data/Chip_58/QualityCheckPassed/reconstruction_NucleiNr0193.png.mat')
    for i in indices:
        d.__getitem__(i, make_plot=True)
    print('test complete')