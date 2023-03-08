import torch
import torch.nn.functional as F
from torch import nn

from modules.dense_motion import DenseMotionNetworkReg
from modules.LadderEncoder import LadderEncoder
from modules.spade import SPADEGenerator
from modules.util import Hourglass, kp2gaussian


def Generator(arch, **kwarg):
    return OcclusionAwareSPADEGenerator(**kwarg, hasid=True)


class OcclusionAwareSPADEGenerator(nn.Module):
    """
    Generator that given source image, source ldmk image and driving ldmk image try to transform image according to movement trajectories
    according to the ldmks.
    """

    def __init__(
        self,
        num_channels,
        block_expansion,
        max_features,
        dense_motion_params=None,
        with_warp_im=False,
        hasid=False,
        with_gaze_htmap=False,
        with_ldmk_line=False,
        with_mouth_line=False,
        with_ht=False,
        ladder=None,
        use_IN=False,
        use_SN=True
    ):
        super(OcclusionAwareSPADEGenerator, self).__init__()
        self.with_warp_im = with_warp_im
        self.with_gaze_htmap = with_gaze_htmap
        self.with_ldmk_line = with_ldmk_line
        self.with_mouth_line = with_mouth_line
        self.with_ht = with_ht
        self.ladder = ladder
        self.use_IN = use_IN
        self.use_SN = use_SN

        ladder_norm_type = "spectralinstance" if use_SN else "instance"
        self.ladder_network = LadderEncoder(**ladder, norm_type=ladder_norm_type)
        self.dense_motion_network = DenseMotionNetworkReg(
            **dense_motion_params
        )

        num_blocks = 3
        self.feature_encoder = Hourglass(
                block_expansion=block_expansion,
                in_features=3,
                max_features=max_features,
                num_blocks=num_blocks,
                Lwarp=False,
                AdaINc=0,
                dec_lease=0,
                use_IN=use_IN
            )
        self.fuse_high_res = nn.Conv2d(block_expansion + 3, block_expansion, kernel_size=(3, 3), padding=(1, 1))

        norm = "spectral" if self.use_SN else ""
        norm += "spadeinstance3x3" if self.use_IN else "spadebatch3x3"
        if hasid:
            norm += "id"
        class_dim = 256
        label_nc_offset = 0  # if with_warp_im else 256
        label_nc_offset = label_nc_offset + 8 if with_gaze_htmap else label_nc_offset
        label_nc_offset = label_nc_offset + 6 if with_ldmk_line else label_nc_offset
        label_nc_offset = label_nc_offset + 3 if with_mouth_line else label_nc_offset
        label_nc_offset = label_nc_offset + 59 if with_ht else label_nc_offset
        label_nc_offset = label_nc_offset + 1  # For occlusion map
        label_nc_list = [512, 512, 512, 512, 256, 128, 64]
        label_nc_list = [ln + label_nc_offset for ln in label_nc_list]
        self.SPDAE_G = SPADEGenerator(
            conv_dim=32,
            label_nc=label_nc_list,
            norm_G=norm,
            class_dim=class_dim,
        )

        self.num_channels = num_channels

        self.apply(self._init_weights)

    def _init_weights(self, m):
        gain = 0.02
        if isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            if m.weight is not None:
                nn.init.xavier_normal_(m.weight, gain=gain)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.xavier_normal_(m.weight, gain=gain)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode="bilinear")
            deformation = deformation.permute(0, 2, 3, 1)
        return F.grid_sample(
            inp.to(deformation.dtype), deformation, padding_mode="reflection"
        )

    def get_gaze_ht(self, source_image, kp_driving):
        spatial_size = source_image.shape[2:]
        gaussian_driving = kp2gaussian(
            kp_driving, spatial_size=spatial_size, kp_variance=0.005
        )
        return gaussian_driving[:, 29:37]

    def forward_warp(
        self,
        source_image,
        ldmk_line=None
    ):
        output_dict = {}

        input_t = (
            source_image
            if ldmk_line is None
            else torch.cat((source_image, ldmk_line), dim=1)
        )
        
        style_feat, _ = self.ladder_network(input_t)
        drv_exp = style_feat
        dense_motion = self.dense_motion_network(input_t, drv_exp)

        output_dict["deformation"] = dense_motion["deformation"]
        output_dict["deformed"] = self.deform_input(
            source_image, dense_motion["deformation"]
        )
        output_dict["occlusion_map"] = dense_motion["occlusion_map"]
        output_dict["prediction"] = output_dict["deformed"]
        output_dict["flow"] = dense_motion["flow"]
        return output_dict

    def foward_refine(
        self,
        source_image,
        src_id,
        ldmk_line,
        mouth_line,
        warp_out,
        kp_driving=None,
    ):
        _, out_list = self.feature_encoder(source_image, return_all=True)
        out_list[-1] = self.fuse_high_res(out_list[-1])
        out_list = [self.deform_input(out, warp_out["deformation"]) for out in out_list]

        feature_list = []
        for out in out_list:
            if self.with_gaze_htmap:
                gaze_htmap = self.get_gaze_ht(out, kp_driving)

            inputs = out
            if out.shape[2] != warp_out["occlusion_map"].shape[2] or out.shape[3] != warp_out["occlusion_map"].shape[3]:
                occlusion_map = F.interpolate(
                    warp_out["occlusion_map"], size=out.shape[2:], mode="bilinear"
                )
            else:
                occlusion_map = warp_out["occlusion_map"]
            inputs = torch.cat((inputs, occlusion_map), dim=1)
            
            if self.with_gaze_htmap:
                inputs = torch.cat((inputs, gaze_htmap), dim=1)
            if self.with_ldmk_line:
                ldmk_line = F.interpolate(ldmk_line, size=inputs.shape[2:], mode="bilinear")
                inputs = torch.cat((inputs, ldmk_line), dim=1)
            if self.with_mouth_line:
                mouth_line = F.interpolate(
                    mouth_line, size=inputs.shape[2:], mode="bilinear"
                )
                inputs = torch.cat((inputs, mouth_line), dim=1)
            
            feature_list.append(inputs)

        outs = self.SPDAE_G(feature_list, class_emb=src_id)

        warp_out["prediction"] = outs
        return warp_out

    def forward(
        self,
        source_image,
        kp_driving=None,
        src_id=None,
        stage=None,
        ldmk_line=None,
        mouth_line=None,
        warp_out=None,
    ):
        if stage == "Warp":
            return self.forward_warp(source_image, ldmk_line)
        elif stage == "Refine":
            return self.foward_refine(
                source_image,
                src_id,
                ldmk_line,
                mouth_line,
                warp_out,
                kp_driving,
            )
        else:
            raise Exception("Unknown stage.")
