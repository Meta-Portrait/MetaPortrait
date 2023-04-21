import os
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn
import torch.nn.functional as F
import torchvision.utils as tvu


class Visualizer(object):
    
    def __init__(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)
        self.writer = SummaryWriter(path)
        self.imgs_path = path.replace('events', 'Imgs')
        if not os.path.isdir(self.imgs_path):
            os.makedirs(self.imgs_path)
        
    def Scalar(self, name, val, step):
        self.writer.add_scalar(name, val, step)
        
    def Image(self, name, val, step):
        x = tvu.make_grid(val)
        self.writer.add_image(name, x, step)

    def save_image(self, val, val_deform, step, srcs, drv, num, src_ldmk_line=None, drv_ldmk_line=None, test_flag=-1):
        src = srcs[:num]
        imgs = torch.cat((drv[:num], drv_ldmk_line[:num], src, src_ldmk_line[:num], val_deform[:num], val[:num]), dim=0)

        if test_flag == -1:
            tvu.save_image(imgs, os.path.join(self.imgs_path, str(step)+'.png'), nrow=num, normalize=True)
        else:
            tvu.save_image(imgs, os.path.join(self.imgs_path, 'test_' + str(step) + '_' + str(test_flag) +'.png'), nrow=num, normalize=True)
