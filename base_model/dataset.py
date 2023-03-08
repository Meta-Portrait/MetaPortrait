import os
import random
import pickle
from PIL import Image
import pathlib

import numpy as np
import cv2
import glob

import torch
from torch.utils import data
from torchvision import transforms

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}


class PersonalDataset(data.Dataset):
    def __init__(self, conf, name, is_train=True):
        
        self.conf = conf
        self.name = name
        self.is_train = is_train
        self.pid = 0

        self.root = conf['dataset'][name]['root']
        self.size = conf['dataset']['frame_shape']

        self.data = {}

        root_path = pathlib.Path(self.root)
        map_dict = pickle.load(open(os.path.join(root_path, 'src_map_dict.pkl'), 'rb'))

        imgs = sorted([file for ext in IMAGE_EXTENSIONS
                        for file in glob.glob(self.root + '/**/[!src_]*.{}'.format(ext), recursive=True)])
        ldmks = sorted([file for file in glob.glob(self.root + '/**/[!src_]*_ldmk.npy', recursive=True)])
        thetas = sorted([file for file in glob.glob(self.root + '/**/[!src_]*_theta.npy', recursive=True)])

        self.data['imgs'] = imgs
        self.data['ldmks'] = ldmks
        self.data['thetas'] = thetas
        self.map_dict = map_dict

        self.dataset_size = len(imgs)

        self.trans_tensor = transforms.Compose([transforms.ToTensor()])
        self.ldmkimg = self.conf['dataset'].get('ldmkimg', False)

        self.num_mask_classes = self.conf['dataset'].get('num_mask_classes', 11)

        with open('./utils/dense_669_connectivity.tsv') as f:
            lines = f.readlines()
            self.connectivity = [(int(m.split(' ')[0]), int(m.split(' ')[1])) for m in lines]

        if self.ldmkimg:
            self.colors = []
            cube_root = len(self.connectivity) ** (1.0 / 3)
            for r in range(0, 255, int(255 / cube_root)):
                for g in range(0, 255, int(255 / cube_root)):
                    for b in range(0, 255, int(255 / cube_root)):
                        self.colors.append([int(r), int(g), int(b)])
            random.seed(0)
            self.colors = self.colors[-len(self.connectivity):]
            random.shuffle(self.colors)
    
    def __len__(self):
        return self.dataset_size
    
    def Init(self):
        self.pid = os.getpid()

    def ldmk_proc(self, ldmk):
        ldmk_norm = ldmk * 2 - 1
        ldmk_norm = ldmk_norm[self.conf['dataset']['ldmk_idx'],:]
        if self.conf['dataset'].get('use_ukt', False) and not self.conf['dataset'].get('eye_enhance', False):
            left_gaze = ldmk_norm[29:33].mean(0, keepdims=True)
            right_gaze = ldmk_norm[33:37].mean(0, keepdims=True)
            ldmk_norm = np.concatenate((ldmk_norm[:29], left_gaze, right_gaze, ldmk_norm[37:]), axis=0)
        return ldmk_norm

    def __getitem__(self, idx):
        if os.getpid() != self.pid:
            self.Init()

        drv_idx = idx
        drv_dir_and_name = self.data['imgs'][idx].split("/")[-3:]
        drv_name = self.data['imgs'][idx].split("/")[-1]
        src_idx = self.map_dict[drv_name]

        src_img_path = os.path.join(self.root, 'src_{}.png'.format(src_idx))
        src_img = np.array(Image.open(src_img_path).resize((self.size[0], self.size[1]), Image.BICUBIC))
        src_img_tensor = self.trans_tensor(src_img.astype(np.float32) / 255)

        drv_img_path = self.data['imgs'][drv_idx]
        drv_img = np.array(Image.open(drv_img_path).resize((self.size[0], self.size[1]), Image.BICUBIC))
        drv_img_tensor = self.trans_tensor(drv_img.astype(np.float32) / 255)

        src_ldmk_path = os.path.join(self.root, 'src_{}_ldmk.npy'.format(src_idx))
        src_ldmk = np.load(src_ldmk_path).astype(np.float32)
        src_ldmk_line = self.draw_kp(None, src_ldmk, (self.size[0], self.size[1]), is_connect=True)
        src_ldmk_line = self.trans_tensor(src_ldmk_line.astype(np.float32) / 255)
        src_ldmk_norm = self.ldmk_proc(src_ldmk)

        drv_ldmk_path = self.data['ldmks'][drv_idx]
        drv_ldmk = np.load(drv_ldmk_path).astype(np.float32)
        if self.conf['dataset'].get('ldmk_jitter', False):
            drv_ldmk = self.ldmk_jitter(drv_ldmk, self.size[0], self.size[1], self.conf['dataset'].get('jitter_sigma', 5)).astype(np.float32)
        drv_ldmk_line = self.draw_kp(None, drv_ldmk, (self.size[0], self.size[1]), is_connect=True)
        drv_ldmk_line = self.trans_tensor(drv_ldmk_line.astype(np.float32) / 255)
        drv_ldmk_norm = self.ldmk_proc(drv_ldmk)

        src_theta_path = os.path.join(self.root, 'src_{}_theta.npy'.format(src_idx))
        src_theta = np.load(src_theta_path).astype(np.float32)

        drv_theta_path = self.data['thetas'][drv_idx]
        drv_theta = np.load(drv_theta_path).astype(np.float32)

        src_id = np.load(os.path.join(self.root, 'src_{}_id.npy'.format(src_idx))).astype(np.float32)
        src_id_tensor = torch.tensor(src_id).squeeze()
                
        out = {'source': src_img_tensor, 'driving': drv_img_tensor, 'source_ldmk': src_ldmk_norm, 'driving_ldmk': drv_ldmk_norm,
            'source_theta': src_theta, 'driving_theta': drv_theta, 'source_line': src_ldmk_line, 'driving_line': drv_ldmk_line,
            'source_id': src_id_tensor, 'driving_name': drv_name, 'driving_dir_and_name': drv_dir_and_name
            }

        if self.conf['dataset'].get('eye_enhance', False):
            out['eye_mask'] = self.get_focal_mask(drv_ldmk_norm, drv_img_tensor)

        if self.conf['dataset'].get('mouth_enhance', False):
            out['mouth_mask'] = self.get_mouth_area(drv_ldmk_norm, drv_img_tensor)
        
        return out
    
    def get_gaze_deformation(self, src_lmdk, drv_ldmk, driving, r):
        (h, w) = (driving.shape[1], driving.shape[2])
        gaze = torch.zeros((h, w, 2))

        left_gaze = int((drv_ldmk[29,0] + 1) / 2 * h), int((drv_ldmk[29,1] + 1) / 2 * w)
        dx, dy = src_lmdk[29] - drv_ldmk[29]
        for i in range(left_gaze[1] - r, left_gaze[1] + r):
            for j in range(left_gaze[0] - r, left_gaze[0] + r):
                if i + dy.item() >= 0 and i + dy.item() < h and j + dx.item() >= 0 and j + dx.item() < w and i >= 0 and i < h and j >= 0 and j < w:
                    gaze[i, j, 0] = dx.item() + 2 * j / w -1
                    gaze[i, j, 1] = dy.item() + 2 * i / h -1

        right_gaze = int((drv_ldmk[30,0] + 1) / 2 * h), int((drv_ldmk[30,1] + 1) / 2 * w)
        dx, dy = src_lmdk[30] - drv_ldmk[30]
        for i in range(right_gaze[1] - r, right_gaze[1] + r):
            for j in range(right_gaze[0] - r, right_gaze[0] + r):
                if i + dy.item() >= 0 and i + dy.item() < h and j + dx.item() >= 0 and j + dx.item() < w and i >= 0 and i < h and j >= 0 and j < w:
                    gaze[i, j, 0] = dx.item() + 2 * j / w -1
                    gaze[i, j, 1] = dy.item() + 2 * i / h -1
        return gaze

    def get_focal_mask(self, drv_ldmk, driving):
        (h, w) = (driving.shape[1], driving.shape[2])
        if self.conf['dataset'].get('focal_loss', 0):
            mask = torch.ones((1, h, w))
            val = self.conf['dataset']['focal_loss']
            ex_ear = (np.concatenate((drv_ldmk[:36], drv_ldmk[38:]), axis=0) + 1) / 2
            minx = int(max(min(ex_ear[:, 0]), 0) * (w - 1))
            maxx = int(min(max(ex_ear[:, 0]), 1) * (w - 1))
            miny = int(max(min(ex_ear[:, 1]), 0) * (h - 1))
            maxy = int(min(max(ex_ear[:, 1]), 1) * (h - 1))
            mask[0, miny:maxy+1, minx:maxx+1] = val
        elif self.conf['dataset'].get('eye_enhance', False):
            mask = torch.zeros((1, h, w))
            left_eye_l, left_eye_r = int((drv_ldmk[1, 0] + 1) * w / 2), int((drv_ldmk[2, 0] + 1) * w / 2)
            left_eye_t, left_eye_d = int((drv_ldmk[3, 1] + 1) * h / 2), int((drv_ldmk[4, 1] + 1) * h / 2)
            right_eye_l, right_eye_r = int((drv_ldmk[5, 0] + 1) * w / 2), int((drv_ldmk[6, 0] + 1) * w / 2)
            right_eye_t, right_eye_d = int((drv_ldmk[7, 1] + 1) * h / 2), int((drv_ldmk[8, 1] + 1) * h / 2)

            pad = 3
            mask[0, max(0, left_eye_t - pad):min(h, left_eye_d + pad + 1), max(0, left_eye_l - pad):min(w, left_eye_r + pad + 1)] = 1
            mask[0, max(0, right_eye_t - pad):min(h, right_eye_d + pad + 1), max(0, right_eye_l - pad):min(w, right_eye_r + pad + 1)] = 1
        return mask

    def get_mouth_area(self, drv_ldmk, driving):
        (h, w) = (driving.shape[1], driving.shape[2])

        mask = torch.zeros((1, h, w))
        mouth_l, mouth_r = int((drv_ldmk[14, 0] + 1) * w / 2), int((drv_ldmk[15, 0] + 1) * w / 2)
        mouth_t, mouth_d = int((drv_ldmk[56, 1] + 1) * h / 2), int((drv_ldmk[57, 1] + 1) * h / 2)

        pad = 5
        mask[0, max(0, mouth_t - pad):min(h, mouth_d + pad + 1), max(0, mouth_l - pad):min(w, mouth_r + pad + 1)] = 1
        return mask

    def draw_kp(self, frame, kp, size=(256,256), is_connect=False, color=(255,255,255)):
        if frame is None:
            if self.ldmkimg:
                frame = np.zeros((size[0], size[1], 3), dtype=np.uint8)
            else:
                frame = np.zeros((size[0], size[1]), dtype=np.uint8)
        else:
            frame = cv2.resize(frame, size)

        if is_connect:
            kp_int = kp.copy()
            for i in range(kp.shape[0]):
                kp_int[i] = [int((kp[i][0]) * size[0]), int((kp[i][1]) * size[1])]
                
            kp_pairs = kp_int[self.connectivity].astype(int)

            if self.ldmkimg:
                for i, (p_0, p_1) in enumerate(kp_pairs):
                    l_c = self.colors[i]
                    cv2.line(frame, tuple(p_0), tuple(p_1), l_c, 1, cv2.LINE_AA)
            else:
                for p_0, p_1 in kp_pairs:
                    cv2.line(frame, tuple(p_0+1), tuple(p_1+1), 0, 1, cv2.LINE_AA)
                for p_0, p_1 in kp_pairs:
                    cv2.line(frame, tuple(p_0), tuple(p_1), color, 1, cv2.LINE_AA)
        else:
            for i in range(kp.shape[0]):
                x = int((kp[i][0]) * size[0])
                y = int((kp[i][1]) * size[1])
                thinkness = 1 if is_connect else 3
                frame = cv2.circle(frame, (x, y), thinkness, color, -1)

        return frame
