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
    opt['phase'] = 'train'

    opt['name'] = 'HDTF'
    opt['type'] = 'HDTFRecurrentDataset'
    if mode == 'folder':
        opt['dataroot_gt'] = 'datasets/data/HDTF_images_100/512'
        opt['dataroot_lq'] = 'datasets/data/HDTF_images_100/512'
        opt['dataroot_flow'] = None
        # opt['meta_info_file'] = 'basicsr/data/meta_info/meta_info_REDS_GT.txt'
        opt['io_backend'] = dict(type='disk')

    degradation_opt = {
            "blur_kernel_size": 5,
            "kernel_list": ['iso', 'aniso'],
            "kernel_prob": [0.1, 0.1],
            "blur_sigma": [0.1, 1],
            "downsample_range": [0.8, 2],
            "noise_range": [0, 5],
            "jpeg_range": [90, 100],
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
    opt['num_frame'] = 5
    opt['gt_size'] = 512
    opt['interval_list'] = [1]
    opt['random_reverse'] = False
    opt['use_hflip'] = True
    opt['use_rot'] = False

    opt['use_shuffle'] = True
    opt['num_worker_per_gpu'] = 1
    opt['batch_size_per_gpu'] = 16
    opt['scale'] = 1

    opt['dataset_enlarge_ratio'] = 1

    os.makedirs('tmp512', exist_ok=True)

    dataset = build_dataset(opt)
    data_loader = build_dataloader(dataset, opt, num_gpu=0, dist=opt['dist'], sampler=None)

    nrow = int(math.sqrt(opt['batch_size_per_gpu']))
    padding = 2 if opt['phase'] == 'train' else 0

    print('start...')
    for i, data in enumerate(data_loader):
        if i > 5:
            break
        print(i)

        lq = data['lq']
        gt = data['gt']
        key = data['key']
        print(key)
        for j in range(opt['num_frame']):
            torchvision.utils.save_image(
                lq[:, j, :, :, :], f'tmp512/lq_{i:03d}_frame{j}.png', nrow=nrow, padding=padding, normalize=False)
            torchvision.utils.save_image(
                gt[:, j, :, :, :], f'tmp512/gt_{i:03d}_frame{j}.png', nrow=nrow, padding=padding, normalize=False)
            # for k in range(opt['batch_size_per_gpu']):
            #     torchvision.utils.save_image(
            #         lq[k, j, :, :, :], f'tmp512/batch/lq_{k:03d}.png', normalize=False)
            # break


if __name__ == '__main__':
    main()
