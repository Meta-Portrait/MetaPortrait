from torch import nn
import torch
import functools
from modules.util import (
    Hourglass,
    make_coordinate_grid,
    LayerNorm2d,
)


class DenseMotionNetworkReg(nn.Module):
    def __init__(
        self,
        block_expansion,
        num_blocks,
        max_features,
        Lwarp=False,
        AdaINc=0,
        dec_lease=0,
        label_nc=0,
        ldmkimg=False,
        occlusion=False,
    ):
        super(DenseMotionNetworkReg, self).__init__()
        in_c = 3 + label_nc + 2 * 3 if ldmkimg else 3 + label_nc
        self.hourglass = Hourglass(
            block_expansion=block_expansion,
            in_features=in_c,
            max_features=max_features,
            num_blocks=num_blocks,
            Lwarp=Lwarp,
            AdaINc=AdaINc,
            dec_lease=dec_lease,
        )

        self.occlusion = occlusion
        if dec_lease > 0:
            norm_layer = functools.partial(LayerNorm2d, affine=True)
            self.reger = nn.Sequential(
                norm_layer(self.hourglass.out_filters),
                nn.LeakyReLU(0.1),
                nn.Conv2d(
                    self.hourglass.out_filters, 2, kernel_size=7, stride=1, padding=3
                ),
            )
            if occlusion:
                self.occlusion_net = nn.Sequential(
                    norm_layer(self.hourglass.out_filters),
                    nn.LeakyReLU(0.1),
                    nn.Conv2d(
                        self.hourglass.out_filters,
                        1,
                        kernel_size=7,
                        stride=1,
                        padding=3,
                    ),
                )
        else:
            self.reger = nn.Conv2d(
                self.hourglass.out_filters, 2, kernel_size=(7, 7), padding=(3, 3)
            )

    def forward(self, source_image, drv_deca):
        prediction = self.hourglass(source_image, drv_exp=drv_deca)

        out_dict = {}
        flow = self.reger(prediction)
        bs, _, h, w = flow.shape
        flow_norm = 2 * torch.cat(
            [flow[:, :1, ...] / (w - 1), flow[:, 1:, ...] / (h - 1)], 1
        )
        out_dict["flow"] = flow_norm
        grid = make_coordinate_grid((h, w), type=torch.FloatTensor).to(flow_norm.device)
        deformation = grid + flow_norm.permute(0, 2, 3, 1)
        out_dict["deformation"] = deformation

        if self.occlusion:
            occlusion_map = torch.sigmoid(self.occlusion_net(prediction))
            _, _, h_old, w_old = occlusion_map.shape
            _, _, h, w = source_image.shape
            if h_old != h or w_old != w:
                occlusion_map = torch.nn.functional.interpolate(
                    occlusion_map, size=(h, w), mode="bilinear", align_corners=False
                )
            out_dict["occlusion_map"] = occlusion_map
        return out_dict
