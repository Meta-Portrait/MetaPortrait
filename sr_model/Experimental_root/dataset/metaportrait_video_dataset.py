import glob
import torch
from os import path as osp
from torch.utils import data as data
from pathlib import Path
import random

from basicsr.data.data_util import duf_downsample, generate_frame_indices, read_img_seq
from basicsr.utils import get_root_logger, scandir
from basicsr.data.video_test_dataset import VideoTestDataset
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.data.transforms import augment, paired_random_crop

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


@DATASET_REGISTRY.register()
class HDTFRecurrentDataset(data.Dataset):
    """HDTF dataset for training recurrent networks.

    The keys are generated from a meta info txt file.
    basicsr/data/meta_info/meta_info_REDS_GT.txt

    path example
    img_lq_path = self.lq_root / clip_name / self.lq_prefix / f'{neighbor:08d}.png'
    img_gt_path = self.gt_root / clip_name / self.gt_prefix / f'{neighbor:08d}.png'

    Each line contains:
    1. subfolder (clip) name; 2. frame number; 3. image shape, separated by
    a white space.
    Examples:
    000 100 (720,1280,3)
    001 100 (720,1280,3)
    ...

    Key examples: "000/00000000"
    GT (gt): Ground-Truth;
    LQ (lq): Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            dataroot_flow (str, optional): Data root path for flow.
            meta_info_file (str): Path for meta information file.
            val_partition (str): Validation partition types. 'REDS4' or
                'official'.
            io_backend (dict): IO backend type and other kwarg.

            num_frame (int): Window size for input frames.
            gt_size (int): Cropped patched size for gt patches.
            interval_list (list): Interval list for temporal augmentation.
            random_reverse (bool): Random reverse input frames.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super(HDTFRecurrentDataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])
        self.lq_prefix = opt.get('lq_prefix', None)
        self.gt_prefix = opt.get('gt_prefix', None)
        self.num_frame = opt['num_frame']
        # if "degradation" in self.opt:
        # self.degradator = DegradationModule(self.opt['degradation'])
        self.keys = []
        # with open(opt['meta_info_file'], 'r') as fin:
        logger = get_root_logger()
        logger.info(f"glob gt with {self.gt_root}"+self.opt.get('gt_temp', "/*/*"))
        gt_list = sorted(glob.glob(f"{self.gt_root}"+self.opt.get('gt_temp', "/*/*")))
        logger.info(f"glob lq with {self.lq_root}"+self.opt.get('lq_temp', "/*/*"))
        lq_list = sorted(glob.glob(f"{self.lq_root}"+self.opt.get('lq_temp', "/*/*")))


        # self.lq_prefix = 'final'
        # self.gt_prefix = 'imgs'

        if opt.get('train_first_frame', None):
            gt_list = gt_list[0:opt.get('train_first_frame', None)]
            lq_list = lq_list[0:opt.get('train_first_frame', None)]
            logger.info(f'Train on only: [{str(gt_list)}];')

        for i, gt_path in enumerate(gt_list):
            clip_name_frame_name = gt_path.replace(str(self.gt_root) + "/", "").replace(".png", "")
            if self.gt_prefix:
              clip_name_frame_name =  clip_name_frame_name.replace(f"/{self.gt_prefix}", "")
            clip_name_frame_name_lq = lq_list[i].replace(str(self.lq_root) + "/", "").replace(".png", "")
            if self.lq_prefix:
                clip_name_frame_name_lq = clip_name_frame_name_lq.replace(f"/{self.lq_prefix}", "")
            self.keys.extend([clip_name_frame_name])
            assert clip_name_frame_name == clip_name_frame_name_lq, f"the gt frame {clip_name_frame_name} \
                                    does not match lq {clip_name_frame_name_lq}"

        # remove the video clips used in validation
        # if opt['val_partition'] == 'REDS4':
        #     val_partition = ['000', '011', '015', '020']
        # elif opt['val_partition'] == 'official':
        #     val_partition = [f'{v:03d}' for v in range(240, 270)]
        # else:
        #     raise ValueError(f'Wrong validation partition {opt["val_partition"]}.'
        #                      f"Supported ones are ['official', 'REDS4'].")
        # if opt['test_mode']:
        #     self.keys = [v for v in self.keys if v.split('/')[0] in val_partition]
        # else:
        #     self.keys = [v for v in self.keys if v.split('/')[0] not in val_partition]

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            if hasattr(self, 'flow_root') and self.flow_root is not None:
                self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root, self.flow_root]
                self.io_backend_opt['client_keys'] = ['lq', 'gt', 'flow']
            else:
                self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
                self.io_backend_opt['client_keys'] = ['lq', 'gt']

        # temporal augmentation configs
        self.interval_list = opt.get('interval_list', [1])
        self.random_reverse = opt.get('random_reverse', False)
        interval_str = ','.join(str(x) for x in self.interval_list)
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'random reverse is {self.random_reverse}.')

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        key = self.keys[index]
        frame_name = key.split('/')[-1]  # key example: 000/00000000
        clip_name = osp.join(*key.split('/')[:-1])
        # determine the neighboring frames
        interval = random.choice(self.interval_list)

        # ensure not exceeding the borders
        start_frame_idx = int(frame_name)
        # Code fo frame index name from 100-0
        # if start_frame_idx > 100 - self.num_frame * interval:
        #     start_frame_idx = random.randint(0, 100 - self.num_frame * interval)


        while not osp.isfile(self.lq_to_path(clip_name, start_frame_idx + self.num_frame * interval)):
        # while not os.path.isfile(lq_to_path(self, clip_name, start_frame_idx + self.num_frame * interval)):
            start_frame_idx -=1



        end_frame_idx = start_frame_idx + self.num_frame * interval

        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        # get the neighboring LQ and GT frames
        img_lqs = []
        img_gts = []
        lq_path_list = []
        gt_path_list = []
        for neighbor in neighbor_list:
            if self.is_lmdb:
                img_lq_path = f'{clip_name}/{neighbor:08d}'
                img_gt_path = f'{clip_name}/{neighbor:08d}'
            else:
                if self.lq_prefix:
                    img_lq_path = self.lq_root / clip_name / self.lq_prefix / f'{neighbor:08d}.png'
                else:
                    img_lq_path = self.lq_root / clip_name / f'{neighbor:08d}.png'
                if self.gt_prefix:
                    img_gt_path = self.gt_root / clip_name / self.gt_prefix / f'{neighbor:08d}.png'
                else:
                    img_gt_path = self.gt_root / clip_name / f'{neighbor:08d}.png'

            # get LQ
            # print(img_lq_path)
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)
            img_lqs.append(img_lq)
            lq_path_list.append(img_lq_path)

            # get GT
            # print(img_gt_path)
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            img_gts.append(img_gt)
            gt_path_list.append(img_gt_path)

        # randomly crop
        img_gts, img_lqs = paired_random_crop(img_gts, img_lqs, gt_size, scale, img_gt_path)

        # add noise to lq
        if hasattr(self, "degradator"):
            img_lqs = self.degradator.degrade(img_lqs)
        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        img_results = augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])

        img_results = img2tensor(img_results)
        img_gts = torch.stack(img_results[len(img_lqs) // 2:], dim=0)
        img_lqs = torch.stack(img_results[:len(img_lqs) // 2], dim=0)

        # img_lqs: (t, c, h, w)
        # img_gts: (t, c, h, w)
        # key: str
        # code for debuging
        # import numpy as np
        # import torchvision.utils as tvu
        # import os
        # os.makedirs( os.path.dirname(f'trash/img_gt{key}.png'), exist_ok=True)
        # tvu.save_image(img_gts, f'trash/img_gt{key}.png',normalize=True)
        # logger = get_root_logger()
        # logger.info(lq_path_list)
        # tvu.save_image(img_lqs, f'trash/img_lq{key}.png',normalize=True)
        # logger.info(gt_path_list)
        return {'lq': img_lqs, 'gt': img_gts, 'key': key}
        # 'lq_path':lq_path_list, 'gt_path':gt_path_list }
    def lq_to_path(self, clip_name, neighbor):
        if self.lq_prefix:
            img_lq_path = self.lq_root / clip_name / self.lq_prefix / f'{neighbor:08d}.png'
        else:
            img_lq_path = self.lq_root / clip_name / f'{neighbor:08d}.png'
        return img_lq_path
    def __len__(self):
        return len(self.keys)