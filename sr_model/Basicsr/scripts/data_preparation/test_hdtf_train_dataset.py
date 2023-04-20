import math
import os
import torchvision.utils

from basicsr.data import build_dataloader, build_dataset


def main(mode='folder'):
    """Test hdtf dataset.

    Args:
        mode: There are two modes: 'lmdb', 'folder'.
    """
    opt = {}
    opt['dist'] = False
    opt['phase'] = 'test'

    opt['name'] = 'HDTF'
    opt['type'] = 'HDTFRecurrentDataset'
    if mode == 'folder':
        opt['dataroot_gt'] = '/mnt/blob/data/HDTF_test_100frames/256'
        opt['dataroot_lq'] = '/mnt/blob/data/HDTF_test_100frames/256'
        opt['test_mode'] = False
        opt['dataroot_flow'] = None
        # opt['meta_info_file'] = 'basicsr/data/meta_info/meta_info_REDS_GT.txt'
        opt['io_backend'] = dict(type='disk')

    degradation_opt = {
            "blur_kernel_size": 11,
            "kernel_list": ['iso', 'aniso'],
            "kernel_prob": [0.2, 0.2],
            "blur_sigma": [0.1, 3],
            "downsample_range": [0.8, 4],
            "noise_range": [0, 5],
            "jpeg_range": [70, 100],
    }
    # degradation_opt = {
    #         "blur_kernel_size": 21,
    #         "kernel_list": ['iso', 'aniso'],
    #         "kernel_prob": [0.5, 0.5],
    #         "blur_sigma": [0.1, 5],
    #         "downsample_range": [0.8, 4],
    #         "noise_range": [0, 10],
    #         "jpeg_range": [60, 100],
    # }
    opt['degradation'] = degradation_opt
    # opt['val_partition'] = 'REDS4'
    opt['num_frame'] = 100
    opt['gt_size'] = 256
    opt['interval_list'] = [1]
    opt['random_reverse'] = False
    opt['use_hflip'] = False
    opt['use_rot'] = False

    opt['use_shuffle'] = False
    opt['num_worker_per_gpu'] = 1
    opt['batch_size_per_gpu'] = 1
    opt['scale'] = 1

    opt['dataset_enlarge_ratio'] = 1

    os.makedirs('256_degrade', exist_ok=True)

    dataset = build_dataset(opt)
    data_loader = build_dataloader(dataset, opt, num_gpu=0, dist=opt['dist'], sampler=None)

    nrow = int(math.sqrt(opt['batch_size_per_gpu']))
    padding = 2 if opt['phase'] == 'train' else 0

    print('start...')
    # for i, data in enumerate(data_loader):
    import numpy as np
    for i in np.arange(10):
        # if i > 5:
            # break
        print(i)
        data = dataset[i*100]
        lq = data['lq']
        gt = data['gt']
        key = data['key']
        print(key)
        os.makedirs(f'256_degrade/{i:03d}', exist_ok=True)
        for j in range(opt['num_frame']):


            torchvision.utils.save_image(
                lq[j, :, :, :], f'256_degrade/{i:03d}/{j:08d}.png', nrow=nrow, padding=padding, normalize=False)
            # torchvision.utils.save_image(
                # gt[j, :, :, :], f'256_degrade/{i:03d}/{j:08d}.png', nrow=nrow, padding=padding, normalize=False)
            # for k in range(opt['batch_size_per_gpu']):
            #     torchvision.utils.save_image(
            #         lq[k, j, :, :, :], f'tmp512/batch/lq_{k:03d}.png', normalize=False)
            # break


if __name__ == '__main__':
    main()
