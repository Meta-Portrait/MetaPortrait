import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm


def get_nonspade_norm_layer(norm_type="instance"):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, "out_channels"):
            return getattr(layer, "out_channels")
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        subnorm_type = norm_type
        if norm_type.startswith("spectral"):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len("spectral") :]

        if subnorm_type == "none" or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, "bias", None) is not None:
            delattr(layer, "bias")
            layer.register_parameter("bias", None)

        if subnorm_type == "batch":
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == "sync_batch":
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == "instance":
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError("normalization layer %s is not recognized" % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer


class LadderEncoder(nn.Module):
    """ Same architecture as the image discriminator """

    def __init__(self, need_feat=False, use_mask=False, label_nc=0, z_dim=512, norm_type="spectralinstance"):
        super().__init__()
        self.need_feat = need_feat
        ldmk_img_nc = 3

        nif = 3 + label_nc + 2 * ldmk_img_nc

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        nef = 64
        norm_layer = get_nonspade_norm_layer(norm_type)
        self.layer1 = norm_layer(nn.Conv2d(nif, nef, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(nef * 1, nef * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(nef * 2, nef * 4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(nef * 4, nef * 8, kw, stride=2, padding=pw))
        self.layer5 = norm_layer(nn.Conv2d(nef * 8, nef * 8, kw, stride=2, padding=pw))
        self.layer6 = norm_layer(nn.Conv2d(nef * 8, nef * 8, kw, stride=2, padding=pw))

        if need_feat:
            self.up_layer2 = norm_layer(
                nn.Conv2d(nef * 2, nef * 2, kw, stride=1, padding=pw)
            )
            self.up_layer3 = nn.Sequential(
                norm_layer(nn.Conv2d(nef * 4, nef * 2, kw, stride=1, padding=pw)),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            )
            self.up_layer4 = nn.Sequential(
                norm_layer(nn.Conv2d(nef * 8, nef * 2, kw, stride=1, padding=pw)),
                nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
            )
            self.up_layer5 = nn.Sequential(
                norm_layer(nn.Conv2d(nef * 8, nef * 2, kw, stride=1, padding=pw)),
                nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True),
            )
            self.up_layer6 = nn.Sequential(
                norm_layer(nn.Conv2d(nef * 8, nef * 2, kw, stride=1, padding=pw)),
                nn.Upsample(scale_factor=16, mode="bilinear", align_corners=True),
            )

        self.actvn = nn.LeakyReLU(0.2, False)
        self.so = s0 = 4
        self.fc = nn.Linear(nef * 8 * s0 * s0, z_dim)

    def forward(self, x):
        features = None
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode="bilinear")

        x = self.layer1(x)
        x = self.layer2(self.actvn(x))
        if self.need_feat:
            features = self.up_layer2(x)
        x = self.layer3(self.actvn(x))
        if self.need_feat:
            features = self.up_layer3(x) + features
        x = self.layer4(self.actvn(x))
        if self.need_feat:
            features = self.up_layer4(x) + features
        x = self.layer5(self.actvn(x))
        if self.need_feat:
            features = self.up_layer5(x) + features
        x = self.layer6(self.actvn(x))
        if self.need_feat:
            features = self.up_layer6(x) + features

        x = self.actvn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x / (x.norm(dim=-1, p=2, keepdim=True) + 1e-5)

        return x, features
