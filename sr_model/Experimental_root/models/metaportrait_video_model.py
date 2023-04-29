import torch
from collections import Counter
from os import path as osp
import os
from torch import distributed as dist
from tqdm import tqdm, trange
import numpy as np
import torchvision.utils as tvu
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.dist_util import get_dist_info
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.video_base_model import VideoBaseModel


@MODEL_REGISTRY.register()
class MetaportraitVideoRecurrentModel(VideoBaseModel):

    def __init__(self, opt):
        super(MetaportraitVideoRecurrentModel, self).__init__(opt)
        if self.is_train:
            self.fix_flow_iter = opt['train'].get('fix_flow')


    def setup_optimizers(self):
        train_opt = self.opt['train']
        flow_lr_mul = train_opt.get('flow_lr_mul', 1)
        recurrent_lr_mul = train_opt.get('recurrent_lr_mul', 0.2)
        logger = get_root_logger()
        logger.info(f'Multiple the learning rate for flow network with {flow_lr_mul}.')
        if flow_lr_mul == 1:
            optim_params = self.net_g.parameters()
        else:  # separate flow params and normal params for different lr
            normal_params = []
            flow_params = []
            recurrent_params = []
            for name, param in self.net_g.named_parameters():
                if 'spynet' in name:
                    flow_params.append(param)
                elif 'recurrent' in name:
                    recurrent_params.append(param)
                else:
                    opt_rm_param = self.opt['train'].get('opt_rm_param', [])
                    fix_flag = False
                    for k in opt_rm_param:
                        if k in name:
                            fix_flag = True
                    if not fix_flag:
                        normal_params.append(param)
                    else:
                        logger.warning(f'Remove Params {name} from optimizer')
                if not param.requires_grad:
                    logger.warning(f'Params {name} will not be optimized, requires_grad=False')
            optim_params = [
                {  # add normal params first
                    'params': normal_params,
                    'lr': train_opt['optim_g']['lr']
                },
                {
                    'params': flow_params,
                    'lr': train_opt['optim_g']['lr'] * flow_lr_mul
                },
                {
                    'params': recurrent_params,
                    'lr': train_opt['optim_g']['lr'] * recurrent_lr_mul
                },
            ]
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def optimize_parameters(self, current_iter):
        if self.fix_flow_iter:
            logger = get_root_logger()
            if current_iter == 1:
                logger.info(f'Fix flow network and feature extractor for {self.fix_flow_iter} iters.')
                for name, param in self.net_g.named_parameters():
                    if 'spynet' in name or 'edvr' in name:
                        param.requires_grad_(False)
            elif current_iter == self.fix_flow_iter:
                for name, param in self.net_g.named_parameters():
                    if 'spynet' in name or 'edvr' in name:
                        param.requires_grad_(True)
                logger.warning('Train parameters in spynet and edvr.')
                # self.net_g.requires_grad_(True)
        if self.opt['train'].get('requires_grad_all_iter', None):
            logger = get_root_logger()
            if current_iter == self.opt['train'].get('requires_grad_all_iter', None):
                # logger = get_root_logger()
                for name, param in self.net_g.named_parameters():
                    opt_rm_param = self.opt['train'].get('opt_rm_param', [])
                    fix_flag = False
                    for k in opt_rm_param:
                        if k in name:
                            fix_flag = True
                    if not fix_flag:
                        param.requires_grad_(True)
                        logger.warning(f'Free Params {name} from optimizer')
                logger.warning('Free all param in net_g.')
        super(MetaportraitVideoRecurrentModel, self).optimize_parameters(current_iter)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset = dataloader.dataset
        dataset_name = dataset.opt['name']
        with_metrics = self.opt['val']['metrics'] is not None and self.opt['val']['metrics']!='None'
        # with_temporal_metrics = self.opt['val']['temp_metrics'] is not None and self.opt['val']['temp_metrics']!='None'
        # initialize self.metric_results
        # It is a dict: {
        #    'folder1': tensor (num_frame x len(metrics)),
        #    'folder2': tensor (num_frame x len(metrics))
        # }
        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {}
                num_frame_each_folder = Counter(dataset.data_info['folder'])
                for folder, num_frame in num_frame_each_folder.items():
                    self.metric_results[folder] = torch.zeros(
                        num_frame, len(self.opt['val']['metrics']), dtype=torch.float32, device='cuda')
            # initialize the best metric results
            self._initialize_best_metric_results(dataset_name)

        # if with_temporal_metrics:
        #     if not hasattr(self, 'with_temporal_metrics'):  # only execute in the first run
        #         self.with_temporal_metrics = {}
        #         # num_frame_each_folder = Counter(dataset.data_info['folder'])
        #         for folder, num_frame in num_frame_each_folder.items():
        #             self.temporal_metric_results[folder] = torch.zeros(
        #                 len(self.opt['val']['metrics']), dtype=torch.float32, device='cuda')
        #             # FIXME wait to implement metrics per flow in the future
        #             # self.metric_results[folder] = torch.zeros(
        #             #     num_frame, len(self.opt['val']['metrics']), dtype=torch.float32, device='cuda')
        #     # initialize the best metric results
        #     self._initialize_best_metric_results(dataset_name)

        # zero self.metric_results
        rank, world_size = get_dist_info()
        if with_metrics:
            for _, tensor in self.metric_results.items():
                tensor.zero_()

        metric_data = dict()
        num_folders = len(dataset)
        num_pad = (world_size - (num_folders % world_size)) % world_size
        if rank == 0:
            pbar = tqdm(total=len(dataset), unit='folder')

        if isinstance(current_iter, str): # at test time
            current_iter = 0
        # Will evaluate (num_folders + num_pad) times, but only the first num_folders results will be recorded.
        # (To avoid wait-dead)
        for i in tqdm(range(rank, num_folders + num_pad, world_size)):
            idx = min(i, num_folders - 1)
            val_data = dataset[idx]
            folder = val_data['folder']
            if not isinstance(folder, int):
                folder = f"{int(i):03d}"
            # compute outputs
            val_data['lq'].unsqueeze_(0)
            val_data['gt'].unsqueeze_(0)
            self.feed_data(val_data)
            val_data['lq'].squeeze_(0)
            val_data['gt'].squeeze_(0)

            self.test()
            visuals = self.get_current_visuals()

            # tentative for out of GPU memory
            del self.lq
            del self.output
            if 'gt' in visuals:
                del self.gt
            torch.cuda.empty_cache()

            if self.center_frame_only:
                visuals['result'] = visuals['result'].unsqueeze(1)
                if 'gt' in visuals:
                    visuals['gt'] = visuals['gt'].unsqueeze(1)

            # evaluate
            if i < num_folders:
                for idx in tqdm(range(visuals['result'].size(1))):
                    result = visuals['result'][0, idx, :, :, :]
                    result_img = tensor2img([result])  # uint8, bgr
                    metric_data['img'] = result_img
                    if 'gt' in visuals:
                        gt = visuals['gt'][0, idx, :, :, :]
                        gt_img = tensor2img([gt])  # uint8, bgr
                        metric_data['img2'] = gt_img

                    if save_img:
                        if self.opt['is_train']:
                            img_path = osp.join(self.opt['path']['visualization'], f"{current_iter:08d}",
                                                dataset_name, folder, f"{idx:08d}_{self.opt['name']}.png")
                            # image name only for REDS dataset
                            imwrite(result_img, img_path)
                            # raise NotImplementedError('saving image is not supported during training.')
                        else:
                            if self.center_frame_only:  # vimeo-90k
                                clip_ = val_data['lq_path'].split('/')[-3]
                                seq_ = val_data['lq_path'].split('/')[-2]
                                name_ = f'{clip_}_{seq_}'
                                img_path = osp.join(self.opt['path']['visualization'], f"{current_iter:08d}", dataset_name, folder,
                                                    f"{name_}_{self.opt['name']}.png")
                            else:  # others
                                img_path = osp.join(self.opt['path']['visualization'], f"{current_iter:08d}", dataset_name, folder,
                                                    f"{idx:08d}_{self.opt['name']}.png")

                            # image name only for REDS dataset
                        imwrite(result_img, img_path)
                        if self.opt['val'].get('save_input', None):
                            if 'gt' in visuals:
                                gt_img_path = osp.join(self.opt['path']['visualization'], f"{current_iter:08d}", dataset_name+'_gt', f"{int(folder):03d}",
                                                    f"{idx:08d}.png")
                                imwrite(gt_img, gt_img_path)

                            lq_img_path = osp.join(self.opt['path']['visualization'], f"{current_iter:08d}", dataset_name+'_lq', f"{int(folder):03d}",
                                                f"{idx:08d}.png")
                            imwrite(tensor2img([visuals['lq'][0, idx, :, :, :]]), lq_img_path)

                    # calculate metrics
                    if with_metrics:
                        for metric_idx, opt_ in enumerate(self.opt['val']['metrics'].values()):
                            if 'temp' not in opt_['type']:
                                result = calculate_metric(metric_data, opt_)
                                self.metric_results[folder][idx, metric_idx] += result
                if with_metrics:
                    for metric_idx, opt_ in enumerate(self.opt['val']['metrics'].values()):
                        gt_folder = osp.join(self.opt['path']['visualization'], f"{current_iter:08d}", dataset_name+'_gt', f"{int(folder):03d}")
                        out_folder = osp.join(self.opt['path']['visualization'], f"{current_iter:08d}", dataset_name, f"{int(folder):03d}")
                        temporal_folder = {
                            "synthesized": out_folder,
                            "flow": gt_folder,
                            "result_dir": out_folder.replace(dataset_name, f"{dataset_name}_flow")
                        }
                        if 'temp' in opt_['type']:
                            result, warp_error_list = calculate_metric(temporal_folder, opt_)
                            # for k, v in warp_error_dict.items():
                            # print(warp_error_list)
                            # print(self.metric_results[folder])
                            for index in range(len(warp_error_list)):
                                self.metric_results[folder][index, metric_idx] += warp_error_list[index]
                                # self.metric_results[folder][idx, metric_idx] += result
                    self.print_metrics_folder( folder, self.metric_results[folder], dataset_name)
                # progress bar
                if rank == 0:
                    for _ in range(world_size):
                        pbar.update(1)
                        pbar.set_description(f'Folder: {folder}')

                if save_img:
                    folder_list = [osp.join(self.opt['path']['visualization'],  f"{current_iter:08d}", dataset_name, folder) ]
                    from basicsr.utils import folder_to_concat_folder, folder_to_video
                    concat_frame_list = folder_to_concat_folder(folder_list)
                    folder_to_video(concat_frame_list, osp.join(self.opt['path']['visualization'],  f"{current_iter:08d}", dataset_name, f"{folder}.mp4") )
                torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        if rank == 0:
            pbar.close()
            # if save_video:
            if save_img:
                folder_list = [osp.join(self.opt['path']['visualization'],  f"{current_iter:08d}", dataset_name) ]
                from basicsr.utils import folder_to_concat_folder, folder_to_video
                concat_frame_list = folder_to_concat_folder(folder_list)
                folder_to_video(concat_frame_list, osp.join(self.opt['path']['visualization'],  f"{current_iter:08d}", dataset_name, f"{dataset_name}.mp4") )
        if with_metrics:
            if self.opt['dist']:
                # collect data among GPUs
                for _, tensor in self.metric_results.items():
                    dist.reduce(tensor, 0)
                dist.barrier()

            if rank == 0:
                self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def print_metrics_folder(self, folder, tensor, dataset_name):
        metric_results_avg = {
            folder: torch.mean(tensor, dim=0).cpu()
            # for (folder, tensor) in self.metric_results.items()
        }
        # total_avg_results is a dict: {
        #    'metric1': float,
        #    'metric2': float
        # }
        total_avg_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        for folder, tensor in metric_results_avg.items():
            for idx, metric in enumerate(total_avg_results.keys()):
                total_avg_results[metric] += metric_results_avg[folder][idx].item()

        # ------------------------------------------ log the metric ------------------------------------------ #
        log_str = f'Validation {dataset_name}\n'
        for metric_idx, (metric, value) in enumerate(total_avg_results.items()):
            log_str += f'\t # {metric}: {value:.4f}'
            for folder, tensor in metric_results_avg.items():
                log_str += f'\t # {folder}: {tensor[metric_idx].item():.4f}'
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)

    def test(self):
        n = self.lq.size(1)
        self.net_g.eval()

        flip_seq = self.opt['val'].get('flip_seq', False)
        self.center_frame_only = self.opt['val'].get('center_frame_only', False)

        if flip_seq:
            self.lq = torch.cat([self.lq, self.lq.flip(1)], dim=1)

        with torch.no_grad():
            if 'temp_psz' not in self.opt['val']:
                self.output = self.net_g(self.lq)
            else:
                # If the video is too long, crop the video into patch
                temp_psz = self.opt['val']['temp_psz']
                self.output_list = []

                num_seg = np.ceil(n / temp_psz).astype(np.int32)
                # num_last_seg_frames = n % temp_psz
                # num_batches = num_seg
                # num_batch_frames = temp_psz
                # num_last_batch_frames = n % temp_psz

                for fridx in trange(num_seg):
                    # global_queue_buffer.set_batch_index(fridx)
                    start, end = fridx * temp_psz, (fridx + 1) *temp_psz
                    end = np.amin([end, n])
                    inframes = self.lq[:, start: end, ...]

                    self.output_list.append(self.net_g(inframes).cpu())
                    torch.cuda.empty_cache()
                    # convert to appropiate type and return
                self.output = torch.cat(self.output_list, dim=1)

        if flip_seq:
            output_1 = self.output[:, :n, :, :, :]
            output_2 = self.output[:, n:, :, :, :].flip(1)
            self.output = 0.5 * (output_1 + output_2)

        if self.center_frame_only:
            self.output = self.output[:, n // 2, :, :, :]

        self.net_g.train()

    def save_during_training(self, current_iter):
        visuals = self.get_current_visuals()

        # tentative for out of GPU memory
        del self.lq
        del self.output
        if 'gt' in visuals:
            del self.gt
        torch.cuda.empty_cache()
        for key in visuals.keys():
            img_path = osp.join(self.opt['path']['visualization'],'training',
                        key, f"{current_iter:08d}_{self.opt['name']}.png")
            if hasattr(self, 'key'):
                logger = get_root_logger()
                logger.info(f'Save key {self.key}')
                # logger.info(f'Save neighbor list {self.neighbor}')

            os.makedirs(os.path.dirname(img_path) , exist_ok=True)
            # image = torch.cat([visuals['lq'][0], visuals['result'][0], visuals['gt'][0]], axis=2)
            tvu.save_image(visuals[key][0], img_path, normalize=False)

