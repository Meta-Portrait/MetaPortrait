import glob
import torch
from os import path as osp
from torch.utils import data as data

from basicsr.data.data_util import duf_downsample, generate_frame_indices, read_img_seq
from basicsr.utils import get_root_logger, scandir
from basicsr.data.video_test_dataset import VideoTestDataset
from basicsr.utils.registry import DATASET_REGISTRY
from tqdm.contrib import tzip
import numpy as np


@DATASET_REGISTRY.register()
class MetaportraitVideoRecurrentTestDataset(VideoTestDataset):
    """Video test dataset for recurrent architectures, which takes LR video
    frames as input and output corresponding HR video frames.

    Args:
        Same as VideoTestDataset.
        Unused opt:
            padding (str): Padding mode.

    """

    def __init__(self, opt):
        super(MetaportraitVideoRecurrentTestDataset, self).__init__(opt)
        # Find unique folder strings
        self.folders = sorted(list(set(self.data_info['folder'])))
        # self.num_frame = self.opt.get('num_frame', -1)
    def __getitem__(self, index):
        folder = self.folders[index]

        if self.cache_data:
            imgs_lq = self.imgs_lq[folder]
            imgs_gt = self.imgs_gt[folder]
        else:
            imgs_lq = read_img_seq(self.imgs_lq[folder])
            imgs_gt = read_img_seq(self.imgs_gt[folder])
            # raise NotImplementedError('Without cache_data is not implemented.')

        # if  self.num_frame > 0:
        #     imgs_lq = imgs_lq[0:self.num_frame]
        #     imgs_gt = imgs_gt[0:self.num_frame]
        return {
            'lq': imgs_lq,
            'gt': imgs_gt,
            'folder': folder,
        }

    def __len__(self):
        return len(self.folders)
