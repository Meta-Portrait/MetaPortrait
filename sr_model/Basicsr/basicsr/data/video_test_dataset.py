import glob
import torch
from os import path as osp
from torch.utils import data as data

from basicsr.data.data_util import duf_downsample, generate_frame_indices, read_img_seq
from basicsr.utils import get_root_logger, scandir
from basicsr.utils.registry import DATASET_REGISTRY
from tqdm.contrib import tzip
import numpy as np
@DATASET_REGISTRY.register()
class VideoTestDataset(data.Dataset):
    """Video test dataset.

    Supported datasets: Vid4, REDS4, REDSofficial.
    More generally, it supports testing dataset with following structures:

    dataroot
    ├── subfolder1
        ├── frame000
        ├── frame001
        ├── ...
    ├── subfolder1
        ├── frame000
        ├── frame001
        ├── ...
    ├── ...

    For testing datasets, there is no need to prepare LMDB files.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            io_backend (dict): IO backend type and other kwarg.
            cache_data (bool): Whether to cache testing datasets.
            name (str): Dataset name.
            meta_info_file (str): The path to the file storing the list of test
                folders. If not provided, all the folders in the dataroot will
                be used.
            num_frame (int): Window size for input frames.
            padding (str): Padding mode.
            val_first_frame (int): -1, validate the first frames
    """

    def __init__(self, opt):
        super(VideoTestDataset, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.gt_root, self.lq_root = opt.get('dataroot_gt', None), opt['dataroot_lq']
        self.data_info = {'lq_path': [], 'gt_path': [], 'folder': [], 'idx': [], 'border': []}
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        assert self.io_backend_opt['type'] != 'lmdb', 'No need to use lmdb during validation/test.'

        logger = get_root_logger()
        logger.info(f'Generate data info for VideoTestDataset - {opt["name"]}')
        self.imgs_lq, self.imgs_gt = {}, {}
        if 'meta_info_file' in opt:
            with open(opt['meta_info_file'], 'r') as fin:
                subfolders = [line.split(' ')[0] for line in fin]
                subfolders_lq = [osp.join(self.lq_root, key) for key in subfolders]
                subfolders_gt = [osp.join(self.gt_root, key) for key in subfolders]
        else:
            if 'gt_temp' in opt:
                subfolders_lq = sorted(glob.glob(osp.join(self.lq_root, opt['lq_temp'])))
                subfolders_gt = sorted(glob.glob(osp.join(self.gt_root, opt['gt_temp'])))
            else:
                subfolders_lq = sorted(glob.glob(osp.join(self.lq_root, '*')))
                subfolders_gt = sorted(glob.glob(osp.join(self.gt_root, '*')))

        # if len(subfolders_lq) != len (subfolders_gt):
        logger.info(f"the number of folder {len(subfolders_lq)} and {len (subfolders_gt)} should be same")


        # logger.info(f'Clean wrong folder')

        if self.opt.get('clean_hdtf_folder', True):
            # Clean wrong folder
            # logger.info(f'Clean wrong folder')
            clean_gt_list = []
            clean_lq_list = []
            for i in np.arange(len(subfolders_lq)):
                lq = subfolders_lq[i]
                gt_for_lq = lq.replace(self.lq_root, self.gt_root).replace('final', 'imgs')
                # logger.info(f'Clean wrong folder')
                if gt_for_lq in subfolders_gt:
                    clean_gt_list.append(gt_for_lq)
                    clean_lq_list.append(lq)
                else:
                    # logger.info(f'Clean wrong folder')
                    # logger = get_root_logger()
                    logger.info(f'Gt {gt_for_lq} for lq {lq} not exist')
                    # print(f'Gt {gt_for_lq} for lq {lq} not exist')
            subfolders_gt = clean_gt_list
            subfolders_lq = clean_lq_list

        assert len(subfolders_lq) == len (subfolders_gt), f"the number of folder should be same {len(subfolders_lq)} != {len(subfolders_gt)}"

        # if opt['name'].lower() in ['vid4', 'reds4', 'redsofficial']:
        for subfolder_lq, subfolder_gt in tzip(subfolders_lq, subfolders_gt):
            # get frame list for lq and gt
            subfolder_lq_list = subfolder_lq.split('/')

            if len(subfolders_lq)==1 or \
                osp.basename(subfolders_lq[0]) != osp.basename(subfolders_lq[1]):
                subfolder_name = osp.basename(subfolder_lq)
            else:
                subfolder_name = subfolder_lq.split('/')[-2]
                assert subfolders_lq[0].split('/')[-2] != subfolders_lq[1].split('/')[-2]
            # subfolder_name = subfolder_lq
            img_paths_lq = sorted(list(scandir(subfolder_lq, full_path=True)))
            img_paths_gt = sorted(list(scandir(subfolder_gt, full_path=True)))
            if self.opt.get('val_first_frame', -1) > 0:
                img_paths_lq = img_paths_lq[0:self.opt.get('val_first_frame', -1)]
                img_paths_gt = img_paths_gt[0:self.opt.get('val_first_frame', -1)]
            max_idx = len(img_paths_lq)
            if max_idx != len(img_paths_gt):
                logger.info(f'Different number of images in lq {subfolder_lq} ({max_idx})'
                                                    f' and gt folders {subfolder_gt} len = ({len(img_paths_gt)})')
                print(f'Different number of images in lq {subfolder_lq} ({max_idx})'
                                                    f' and gt folders {subfolder_gt} len = ({len(img_paths_gt)})')
                continue
            if len(self.data_info['lq_path']) == self.opt.get('val_first_all_frame', -1):
                break
            self.data_info['lq_path'].extend(img_paths_lq)
            self.data_info['gt_path'].extend(img_paths_gt)
            self.data_info['folder'].extend([subfolder_name] * max_idx)
            for i in range(max_idx):
                self.data_info['idx'].append(f'{i}/{max_idx}')
            border_l = [0] * max_idx
            for i in range(self.opt['num_frame'] // 2):
                border_l[i] = 1
                border_l[max_idx - i - 1] = 1
            self.data_info['border'].extend(border_l)

            # cache data or save the frame list
            if self.cache_data:
                logger.info(f'Cache {subfolder_name} for VideoTestDataset...')
                self.imgs_lq[subfolder_name] = read_img_seq(img_paths_lq)
                self.imgs_gt[subfolder_name] = read_img_seq(img_paths_gt)
            else:
                self.imgs_lq[subfolder_name] = img_paths_lq
                self.imgs_gt[subfolder_name] = img_paths_gt
        # else:
        #     raise ValueError(f'Non-supported video test dataset: {type(opt["name"])}')


    def __getitem__(self, index):
        folder = self.data_info['folder'][index]
        idx, max_idx = self.data_info['idx'][index].split('/')
        idx, max_idx = int(idx), int(max_idx)
        border = self.data_info['border'][index]
        lq_path = self.data_info['lq_path'][index]

        select_idx = generate_frame_indices(idx, max_idx, self.opt['num_frame'], padding=self.opt['padding'])

        if self.cache_data:
            imgs_lq = self.imgs_lq[folder].index_select(0, torch.LongTensor(select_idx))
            img_gt = self.imgs_gt[folder][idx]
        else:
            img_paths_lq = [self.imgs_lq[folder][i] for i in select_idx]
            imgs_lq = read_img_seq(img_paths_lq)
            img_gt = read_img_seq([self.imgs_gt[folder][idx]])
            img_gt.squeeze_(0)

        return {
            'lq': imgs_lq,  # (t, c, h, w)
            'gt': img_gt,  # (c, h, w)
            'folder': folder,  # folder name
            'idx': self.data_info['idx'][index],  # e.g., 0/99
            'border': border,  # 1 for border, 0 for non-border
            'lq_path': lq_path  # center frame
        }

    def __len__(self):
        return len(self.data_info['gt_path'])


@DATASET_REGISTRY.register()
class VideoTestVimeo90KDataset(data.Dataset):
    """Video test dataset for Vimeo90k-Test dataset.

    It only keeps the center frame for testing.
    For testing datasets, there is no need to prepare LMDB files.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            io_backend (dict): IO backend type and other kwarg.
            cache_data (bool): Whether to cache testing datasets.
            name (str): Dataset name.
            meta_info_file (str): The path to the file storing the list of test
                folders. If not provided, all the folders in the dataroot will
                be used.
            num_frame (int): Window size for input frames.
            padding (str): Padding mode.
    """

    def __init__(self, opt):
        super(VideoTestVimeo90KDataset, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        if self.cache_data:
            raise NotImplementedError('cache_data in Vimeo90K-Test dataset is not implemented.')
        self.gt_root, self.lq_root = opt['dataroot_gt'], opt['dataroot_lq']
        self.data_info = {'lq_path': [], 'gt_path': [], 'folder': [], 'idx': [], 'border': []}
        neighbor_list = [i + (9 - opt['num_frame']) // 2 for i in range(opt['num_frame'])]

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        assert self.io_backend_opt['type'] != 'lmdb', 'No need to use lmdb during validation/test.'

        logger = get_root_logger()
        logger.info(f'Generate data info for VideoTestDataset - {opt["name"]}')
        with open(opt['meta_info_file'], 'r') as fin:
            subfolders = [line.split(' ')[0] for line in fin]
        for idx, subfolder in enumerate(subfolders):
            gt_path = osp.join(self.gt_root, subfolder, 'im4.png')
            self.data_info['gt_path'].append(gt_path)
            lq_paths = [osp.join(self.lq_root, subfolder, f'im{i}.png') for i in neighbor_list]
            self.data_info['lq_path'].append(lq_paths)
            self.data_info['folder'].append('vimeo90k')
            self.data_info['idx'].append(f'{idx}/{len(subfolders)}')
            self.data_info['border'].append(0)

    def __getitem__(self, index):
        lq_path = self.data_info['lq_path'][index]
        gt_path = self.data_info['gt_path'][index]
        imgs_lq = read_img_seq(lq_path)
        img_gt = read_img_seq([gt_path])
        img_gt.squeeze_(0)

        return {
            'lq': imgs_lq,  # (t, c, h, w)
            'gt': img_gt,  # (c, h, w)
            'folder': self.data_info['folder'][index],  # folder name
            'idx': self.data_info['idx'][index],  # e.g., 0/843
            'border': self.data_info['border'][index],  # 0 for non-border
            'lq_path': lq_path[self.opt['num_frame'] // 2]  # center frame
        }

    def __len__(self):
        return len(self.data_info['gt_path'])


@DATASET_REGISTRY.register()
class VideoTestDUFDataset(VideoTestDataset):
    """ Video test dataset for DUF dataset.

    Args:
        opt (dict): Config for train dataset.
            Most of keys are the same as VideoTestDataset.
            It has the following extra keys:

            use_duf_downsampling (bool): Whether to use duf downsampling to
                generate low-resolution frames.
            scale (bool): Scale, which will be added automatically.
    """

    def __getitem__(self, index):
        folder = self.data_info['folder'][index]
        idx, max_idx = self.data_info['idx'][index].split('/')
        idx, max_idx = int(idx), int(max_idx)
        border = self.data_info['border'][index]
        lq_path = self.data_info['lq_path'][index]

        select_idx = generate_frame_indices(idx, max_idx, self.opt['num_frame'], padding=self.opt['padding'])

        if self.cache_data:
            if self.opt['use_duf_downsampling']:
                # read imgs_gt to generate low-resolution frames
                imgs_lq = self.imgs_gt[folder].index_select(0, torch.LongTensor(select_idx))
                imgs_lq = duf_downsample(imgs_lq, kernel_size=13, scale=self.opt['scale'])
            else:
                imgs_lq = self.imgs_lq[folder].index_select(0, torch.LongTensor(select_idx))
            img_gt = self.imgs_gt[folder][idx]
        else:
            if self.opt['use_duf_downsampling']:
                img_paths_lq = [self.imgs_gt[folder][i] for i in select_idx]
                # read imgs_gt to generate low-resolution frames
                imgs_lq = read_img_seq(img_paths_lq, require_mod_crop=True, scale=self.opt['scale'])
                imgs_lq = duf_downsample(imgs_lq, kernel_size=13, scale=self.opt['scale'])
            else:
                img_paths_lq = [self.imgs_lq[folder][i] for i in select_idx]
                imgs_lq = read_img_seq(img_paths_lq)
            img_gt = read_img_seq([self.imgs_gt[folder][idx]], require_mod_crop=True, scale=self.opt['scale'])
            img_gt.squeeze_(0)

        return {
            'lq': imgs_lq,  # (t, c, h, w)
            'gt': img_gt,  # (c, h, w)
            'folder': folder,  # folder name
            'idx': self.data_info['idx'][index],  # e.g., 0/99
            'border': border,  # 1 for border, 0 for non-border
            'lq_path': lq_path  # center frame
        }


@DATASET_REGISTRY.register()
class VideoRecurrentTestDataset(VideoTestDataset):
    """Video test dataset for recurrent architectures, which takes LR video
    frames as input and output corresponding HR video frames.

    Args:
        Same as VideoTestDataset.
        Unused opt:
            padding (str): Padding mode.

    """

    def __init__(self, opt):
        super(VideoRecurrentTestDataset, self).__init__(opt)
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
