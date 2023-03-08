import os
import datetime
import random

import numpy as np
import cv2
import torch


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
