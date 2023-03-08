from torch import nn
import torch.nn.functional as F
from modules.util import kp2gaussian
import torch
from torch.nn.utils import spectral_norm


class DownBlock2d(nn.Module):
    """
    Simple block for processing video (encoder).
    """

    def __init__(
        self, in_features, out_features, norm=False, kernel_size=4, pool=False, sn=False
    ):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_features, out_channels=out_features, kernel_size=kernel_size
        )

        if sn:
            self.conv = nn.utils.spectral_norm(self.conv)

        if norm:
            self.norm = nn.InstanceNorm2d(out_features, affine=True)
        else:
            self.norm = None
        self.pool = pool

    def forward(self, x):
        out = x
        out = self.conv(out)
        if self.norm:
            out = self.norm(out)
        out = F.leaky_relu(out, 0.2)
        if self.pool:
            out = F.avg_pool2d(out, (2, 2))
        return out


class Discriminator(nn.Module):
    """
    Discriminator similar to Pix2Pix
    """

    def __init__(
        self,
        num_channels=3,
        block_expansion=64,
        num_blocks=4,
        max_features=512,
        sn=False,
        use_kp=False,
        num_kp=10,
        kp_variance=0.01,
        AdaINc=0,
        **kwargs
    ):
        super(Discriminator, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(
                DownBlock2d(
                    num_channels + num_kp * use_kp
                    if i == 0
                    else min(max_features, block_expansion * (2 ** i)),
                    min(max_features, block_expansion * (2 ** (i + 1))),
                    norm=(i != 0),
                    kernel_size=4,
                    pool=(i != num_blocks - 1),
                    sn=sn,
                )
            )

        self.down_blocks = nn.ModuleList(down_blocks)
        self.conv = nn.Conv2d(
            self.down_blocks[-1].conv.out_channels, out_channels=1, kernel_size=1
        )
        if sn:
            self.conv = nn.utils.spectral_norm(self.conv)
        self.use_kp = use_kp
        self.kp_variance = kp_variance

        self.AdaINc = AdaINc
        if AdaINc > 0:
            self.to_exp = nn.Sequential(
                nn.Linear(block_expansion * (2 ** num_blocks), 256),
                nn.LeakyReLU(256),
                nn.Linear(256, AdaINc),
            )

    def forward(self, x, kp=None):
        feature_maps = []
        out = x
        if self.use_kp:
            heatmap = kp2gaussian(kp, x.shape[2:], self.kp_variance)
            out = torch.cat([out, heatmap], dim=1)

        for down_block in self.down_blocks:
            feature_maps.append(down_block(out))
            out = feature_maps[-1]
        prediction_map = self.conv(out)

        if self.AdaINc > 0:
            feat = F.adaptive_avg_pool2d(out, 1)
            exp = self.to_exp(feat.squeeze(-1).squeeze(-1))
        else:
            exp = None

        return feature_maps, prediction_map, exp


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale (scale) discriminator
    """

    def __init__(self, scales=(), **kwargs):
        super(MultiScaleDiscriminator, self).__init__()
        self.scales = scales
        discs = {}
        self.use_kp = kwargs["use_kp"]
        for scale in scales:
            discs[str(scale).replace(".", "-")] = Discriminator(**kwargs)
        self.discs = nn.ModuleDict(discs)
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

    def forward(self, x, kp=None):
        out_dict = {}
        for scale, disc in self.discs.items():
            scale = str(scale).replace("-", ".")
            key = "prediction_" + scale
            feature_maps, prediction_map, exp = disc(x[key], kp)
            out_dict["feature_maps_" + scale] = feature_maps
            out_dict["prediction_map_" + scale] = prediction_map
            out_dict["exp_" + scale] = exp
        return out_dict
