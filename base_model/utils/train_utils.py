import os
import datetime
import random

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import distributed as dist

from modules.generator import Generator
from modules.model import (
    GeneratorFullModel,
    DiscriminatorFullModel
)


def adjust_lr(optim, epoch, args):

    decay_times = len([x for x in args["lr_milestones"] if x - 1 < epoch])
    adj_lr = args["lr"] * ((1 / 10) ** decay_times)
    for pg in optim.param_groups:
        pg["lr"] = adj_lr
        print("learning rate is {}".format(adj_lr))

    return adj_lr


def make_weights(train_data_list, weight_lst=None):

    weights = []
    for i, train_data in enumerate(train_data_list):

        count = len(train_data)
        for j in range(count):
            if weight_lst is not None:
                weights.append(1 / count * weight_lst[i])
            else:
                weights.append(1 / count)

    return weights


def frozen_layers(model, key_list=None):

    for name, param in model.named_parameters():

        if key_list is None:
            param.requires_grad = False
            print("FIX : {}".format(name))
        else:
            for k in key_list:
                if k in name:
                    param.requires_grad = False
                    print("FIX : {}".format(name))
                    break

    return model


def set_random_seed(seed):

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def save_ckpt(out_path, epoch, models, total_iters=None):

    ckpt = {k: v.state_dict() for k, v in models.items()}
    ckpt["epoch"] = epoch

    if out_path.lower().endswith((".pth", ".pth.tar")):
        save_path = out_path

    else:
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if total_iters is not None:
            save_path = os.path.join(out_path, "ckpt_{}_{}.pth.tar".format(epoch, total_iters))
        else:
            save_path = os.path.join(out_path, "ckpt_{}_{}.pth.tar".format(epoch, time))

    torch.save(ckpt, save_path)

    return save_path


def load_ckpt(ckpt_path, models, device=None, strict=True, warp_ckpt=False):
    ckpt = torch.load(ckpt_path, map_location=device)
    epoch = ckpt["epoch"] if "epoch" in ckpt else 0

    for key in models:
        if key in ckpt:
            print("load {} in checkpoint".format(key))
            if not (isinstance(models[key], torch.optim.Adam)):
                if warp_ckpt:
                    pretrained_dict = {k: v for k, v in ckpt[key].items() if 'ladder_network' in k or 'dense_motion_network' in k}
                    model_dict = models[key].state_dict()
                    model_dict.update(pretrained_dict)
                    models[key].load_state_dict(model_dict)
                else:
                    msg = models[key].load_state_dict(ckpt[key], strict=strict)
                    print(msg)
            else:
                models[key].load_state_dict(ckpt[key])

    return epoch


def frame_cropping(frame, bbox=None, expand_ratio=1.0, offset_x=0, offset_y=0):

    img_h = frame.shape[0]
    img_w = frame.shape[1]
    min_img_sz = min(img_w, img_h)
    if bbox is None:
        bbox = [0, 0, 1, 1]
    bbox_w = max(bbox[2] * img_w, bbox[3] * img_h)
    center = [
        int((bbox[0] + bbox[2] * 0.5 + bbox[2] * offset_x) * img_w),
        int((bbox[1] + bbox[3] * 0.5 - bbox[3] * offset_y) * img_h),
    ]

    crop_sz = min(min_img_sz, int(bbox_w * expand_ratio))
    half_sz = int(crop_sz * 0.5)

    if center[0] + half_sz > img_w:
        center[0] = img_w - half_sz
    if center[0] - half_sz < 0:
        center[0] = half_sz
    if center[1] + half_sz > img_h:
        center[1] = img_h - half_sz
    if center[1] - half_sz < 0:
        center[1] = half_sz

    frame_crop = frame[center[1] - half_sz : center[1] + half_sz, center[0] - half_sz : center[0] + half_sz]
    bbox_crop = [(center[0] - half_sz) / img_w, (center[1] - half_sz) / img_h, crop_sz / img_w, crop_sz / img_h]

    return frame_crop, bbox_crop


def ldmk_norm_by_bbox(ldmk, bbox, is_inverse=False):

    ldmk_new = np.zeros(ldmk.shape, dtype=ldmk.dtype)

    if not is_inverse:
        ldmk_new[:, 0] = (ldmk[:, 0] - bbox[0]) / bbox[2]
        ldmk_new[:, 1] = (ldmk[:, 1] - bbox[1]) / bbox[3]

    else:
        ldmk_new[:, 0] = ldmk[:, 0] * bbox[2] + bbox[0]
        ldmk_new[:, 1] = ldmk[:, 1] * bbox[3] + bbox[1]

    return ldmk_new


def draw_kp(frame, kp, size, color=(255, 255, 255)):

    if frame is None:
        frame = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    else:
        frame = cv2.resize(frame, size)
    for i in range(kp.shape[0]):
        x = int((kp[i][0]) * size[0])
        y = int((kp[i][1]) * size[1])
        frame = cv2.circle(frame, (x, y), 1, color, -1)
    return frame

def nan_to_num(input, nan=0.0, posinf=None, neginf=None, *, out=None): # pylint: disable=redefined-builtin
    assert isinstance(input, torch.Tensor)
    if posinf is None:
        posinf = torch.finfo(input.dtype).max
    if neginf is None:
        neginf = torch.finfo(input.dtype).min
    assert nan == 0
    return torch.clamp(input.unsqueeze(0).nansum(0), min=neginf, max=posinf, out=out)


def build_outer_optimizer_and_scheduler(conf, G, D):
    base_params_id = list(map(id, G.dense_motion_network.parameters()))
    if conf["model"]["generator"].get("ladder", False):
        base_params_id += list(map(id, G.ladder_network.parameters()))
    warp_params = filter(lambda p: id(p) in base_params_id, G.parameters())
    refine_params = filter(lambda p: id(p) not in base_params_id, G.parameters())
    optim_G = torch.optim.Adam(
        [
            {"params": refine_params},
            {
                "params": warp_params,
                "lr": conf["train"]["lr_generator"] * conf["train"].get("warplr_tune", 1.0),
            },
        ],
        lr=conf["train"]["lr_generator"],
        betas=(conf["train"].get("outer_beta_1", 0.5), conf["train"].get("outer_beta_2", 0.999)),
    )
    optim_D = torch.optim.Adam(
        D.parameters(), lr=conf["train"]["lr_discriminator"], betas=(conf["train"].get("outer_beta_1", 0.5), conf["train"].get("outer_beta_2", 0.999))
    )

    scheduler_G = MultiStepLR(
        optim_G,
        conf["train"]["epoch_milestones"],
        gamma=0.1,
        last_epoch=-1,
    )
    scheduler_D = MultiStepLR(
        optim_D,
        conf["train"]["epoch_milestones"],
        gamma=0.1,
        last_epoch=-1,
    )
    return optim_G, optim_D, scheduler_G, scheduler_D


def build_inner_optimizer(conf, G_full_clone, D_full_clone):
    base_params_id = list(map(id, G_full_clone.generator.dense_motion_network.parameters()))
    if conf["model"]["generator"].get("ladder", False):
        base_params_id += list(map(id, G_full_clone.generator.ladder_network.parameters()))
    warp_params = filter(lambda p: id(p) in base_params_id, G_full_clone.generator.parameters())
    refine_params = filter(lambda p: id(p) not in base_params_id, G_full_clone.generator.parameters())
    inner_optimizer_G = torch.optim.Adam(
                                    [
                                        {"params": refine_params},
                                        {
                                            "params": warp_params,
                                            "lr": conf["train"]["inner_lr_generator"] * conf["train"].get("inner_warplr_tune", 1.0),
                                        },
                                    ],
                                    lr=conf["train"]["inner_lr_generator"],
                                    betas=(conf["train"].get("inner_beta_1", 0.5), conf["train"].get("inner_beta_2", 0.999)),
                                )
    inner_optimizer_D = torch.optim.Adam(
        D_full_clone.discriminator.parameters(), lr=conf["train"]["inner_lr_discriminator"], betas=(conf["train"].get("inner_beta_1", 0.5), conf["train"].get("inner_beta_2", 0.999))
    )
    return inner_optimizer_G, inner_optimizer_D


def build_full_model(conf, args, G, D):
    G_full = GeneratorFullModel(
        None,
        G,
        D,
        conf["train"],
        conf["model"].get("arch", None),
        rank=args["local_rank"],
        conf=conf,
    )
    if conf["model"]["discriminator"].get("type", "MultiPatchGan") == "MultiPatchGan":
        D_full = DiscriminatorFullModel(None, G, D, conf["train"])
    else:
        raise Exception("Unsupported discriminator type: {}".format(conf["model"]["discriminator"].get("type", "MultiPatchGan")))

    # w/ sync-batchnorm at modules/util.py
    G_full.generator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(G_full.generator)
    G_full = DDP(G_full, device_ids=[args["local_rank"]], find_unused_parameters=True)
    D_full = DDP(
        D_full,
        device_ids=[args["local_rank"]],
        find_unused_parameters=True,
        broadcast_buffers=False,
    )
    return G_full, D_full


def clone_model(conf, args, G, D):
    G_full_clone = GeneratorFullModel(
        None,
        G,
        D,
        conf["train"],
        conf["model"].get("arch", None),
        rank=args["local_rank"],
        conf=conf,
    )
    G_full_clone.cuda()
    D_full_clone = DiscriminatorFullModel(None, G, D, conf["train"])
    D_full_clone.cuda()
    
    return G_full_clone, D_full_clone


def convert_data_to_cuda(data):
    for key, value in data.items():
        if isinstance(value, list):
            if isinstance(value[0], (str, list)):
                continue
            data[key] = [v.cuda() for v in value]
        elif isinstance(value, str):
            continue
        else:
            data[key] = value.cuda()
    return data


def return_batch_data(data, start, end):
    batch_data = {}
    for key, value in data.items():
        batch_data[key] = value[start:end, ...]
    return batch_data


def return_cls_data(data, index):
    cls_data = {}
    for key, value in data.items():
        cls_data[key] = value[:, index, ...]
    return cls_data


def process_vis_data(data):
    for key, value in data.items():
        if isinstance(value, list):
            if isinstance(value[0], str):
                continue
            data[key] = [v.cpu() for v in value]
        elif isinstance(value, str):
            continue
        else:
            data[key] = value.cpu().transpose(0, 1).reshape(-1, *value.shape[2:])
    return data


def save_events(writer, losses_G, loss_G, step, losses_D=None, loss_D=None, generated=None, loss_G_init=None, loss_D_init=None):
    for key in losses_G:
        writer.Scalar(key + "_G", losses_G[key].mean(), step)
    if losses_D is not None:
        for key in losses_D:
            writer.Scalar(key + "_D", losses_D[key].mean(), step)
        writer.Scalar("loss_D", loss_D, step)
    writer.Scalar("loss_G", loss_G, step)
    if loss_D_init is not None:
        writer.Scalar("loss_D_init", loss_D_init, step)
    if loss_G_init is not None:
        writer.Scalar("loss_G_init", loss_G_init, step)
    if generated is not None:
        writer.Image("Pred", generated["prediction"][:16], step)


def save_training_images(conf, args, writer, generated, data, step, num):
    writer.save_image(
                generated["prediction"].cpu(),
                generated["deformed"].cpu(),
                step,
                data["source"],
                data["driving"],
                num=num,
                src_ldmk_line=data["source_line"],
                drv_ldmk_line=data["driving_line"],
            )


def reduce_loss_dict(loss_dict, world_size):
    if world_size < 2:
        return loss_dict

    with torch.no_grad():
        keys = []
        losses = []

        for k in sorted(loss_dict.keys()):
            keys.append(k)
            losses.append(loss_dict[k])

        losses = torch.stack(losses, 0)
        dist.reduce(losses, dst=0)

        if dist.get_rank() == 0:
            losses /= world_size

        reduced_losses = {k: v for k, v in zip(keys, losses)}

    return reduced_losses


def reduce_loss(loss, world_size):
    if world_size < 2:
        return loss

    with torch.no_grad():
        dist.reduce(loss, dst=0) 
        return loss / world_size
