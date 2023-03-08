from torch import nn

import torch.nn.functional as F
import torch

from torch.nn import BatchNorm2d


def kp2gaussian(kp, spatial_size, kp_variance):
    """
    Transform a keypoint into gaussian like representation
    """
    mean = kp["value"]
    coordinate_grid = make_coordinate_grid(spatial_size, mean.type())
    number_of_leading_dimensions = len(mean.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    coordinate_grid = coordinate_grid.view(*shape)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)

    # Preprocess kp shape
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 2)
    mean = mean.view(*shape)

    mean_sub = coordinate_grid - mean
    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)
    return out


def make_coordinate_grid(spatial_size, type):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)

    x = 2 * (x / (w - 1)) - 1
    y = 2 * (y / (h - 1)) - 1

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)

    return meshed


class ResBlock2d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock2d, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_features,
            out_channels=in_features,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.conv2 = nn.Conv2d(
            in_channels=in_features,
            out_channels=in_features,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.norm1 = BatchNorm2d(in_features, affine=True)
        self.norm2 = BatchNorm2d(in_features, affine=True)

    def forward(self, x):
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out


class UpBlock2d(nn.Module):
    """
    Upsampling block for use in decoder.
    """

    def __init__(
        self,
        in_features,
        out_features,
        kernel_size=3,
        padding=1,
        groups=1,
        Lwarp=False,
        AdaINc=0,
        use_IN=False
    ):
        super(UpBlock2d, self).__init__()
        self.AdaINc = AdaINc
        self.conv = nn.Conv2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
        )
        if AdaINc > 0:
            self.norm = ADAIN(out_features, feature_nc=AdaINc)
        elif use_IN:
            self.norm = nn.InstanceNorm2d(out_features, affine=True)
        else:
            self.norm = BatchNorm2d(out_features, affine=True)

        self.Lwarp = Lwarp
        if Lwarp:
            self.SameBlock2d = SameBlock2d(
                out_features, out_features, groups, kernel_size, padding, AdaINc=AdaINc
            )

    def forward(self, x, drv_exp=None):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        if self.AdaINc > 0:
            out = self.norm(out, drv_exp)
        else:
            out = self.norm(out)
        out = F.relu(out)
        if self.Lwarp:
            out = self.SameBlock2d(out, drv_exp=drv_exp)
        return out


class DownBlock2d(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(
        self,
        in_features,
        out_features,
        kernel_size=3,
        padding=1,
        groups=1,
        Lwarp=False,
        AdaINc=0,
        use_IN=False
    ):
        super(DownBlock2d, self).__init__()
        self.AdaINc = AdaINc
        self.conv = nn.Conv2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
        )
        if AdaINc > 0:
            self.norm = ADAIN(out_features, feature_nc=AdaINc)
        elif use_IN:
            self.norm = nn.InstanceNorm2d(out_features, affine=True)
        else:
            self.norm = BatchNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

        self.Lwarp = Lwarp
        if Lwarp:
            self.SameBlock2d = SameBlock2d(
                out_features, out_features, groups, kernel_size, padding, AdaINc=AdaINc
            )

    def forward(self, x, drv_exp=None):
        out = self.conv(x)
        if self.AdaINc > 0:
            out = self.norm(out, drv_exp)
        else:
            out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        if self.Lwarp:
            out = self.SameBlock2d(out, drv_exp=drv_exp)
        return out


class SameBlock2d(nn.Module):
    """
    Simple block, preserve spatial resolution.
    """

    def __init__(
        self, in_features, out_features, groups=1, kernel_size=3, padding=1, AdaINc=0, use_IN=False
    ):
        super(SameBlock2d, self).__init__()
        self.AdaINc = AdaINc
        self.conv = nn.Conv2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
        )
        if AdaINc > 0:
            self.norm = ADAIN(out_features, feature_nc=AdaINc)
        elif use_IN:
            self.norm = nn.InstanceNorm2d(out_features, affine=True)
        else:
            self.norm = BatchNorm2d(out_features, affine=True)

    def forward(self, x, drv_exp=None):
        out = self.conv(x)
        if self.AdaINc > 0:
            out = self.norm(out, drv_exp)
        else:
            out = self.norm(out)
        out = F.relu(out)
        return out


class ADAIN(nn.Module):
    def __init__(self, norm_nc, feature_nc):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)

        nhidden = 128
        use_bias = True

        self.mlp_shared = nn.Sequential(
            nn.Linear(feature_nc, nhidden, bias=use_bias), nn.ReLU()
        )
        self.mlp_gamma = nn.Linear(nhidden, norm_nc, bias=use_bias)
        self.mlp_beta = nn.Linear(nhidden, norm_nc, bias=use_bias)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x, feature):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on feature
        feature = feature.view(feature.size(0), -1)
        actv = self.mlp_shared(feature)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        gamma = gamma.view(*gamma.size()[:2], 1, 1)
        beta = beta.view(*beta.size()[:2], 1, 1)
        out = normalized * (1 + gamma) + beta
        return out


class Encoder(nn.Module):
    """
    Hourglass Encoder
    """

    def __init__(
        self,
        block_expansion,
        in_features,
        num_blocks=3,
        max_features=256,
        Lwarp=False,
        AdaINc=0,
        use_IN=False
    ):
        super(Encoder, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(
                DownBlock2d(
                    in_features
                    if i == 0
                    else min(max_features, block_expansion * (2 ** i)),
                    min(max_features, block_expansion * (2 ** (i + 1))),
                    kernel_size=3,
                    padding=1,
                    Lwarp=Lwarp,
                    AdaINc=AdaINc,
                    use_IN=use_IN
                )
            )
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x, drv_exp=None):
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1], drv_exp=drv_exp))
        return outs


class Decoder(nn.Module):
    """
    Hourglass Decoder
    """

    def __init__(
        self,
        block_expansion,
        in_features,
        num_blocks=3,
        dec_lease=0,
        max_features=256,
        Lwarp=False,
        AdaINc=0,
        use_IN=False
    ):
        super(Decoder, self).__init__()

        up_blocks = []

        for i in range(dec_lease, num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(
                max_features, block_expansion * (2 ** (i + 1))
            )
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(
                UpBlock2d(
                    in_filters,
                    out_filters,
                    kernel_size=3,
                    padding=1,
                    Lwarp=Lwarp,
                    AdaINc=AdaINc,
                    use_IN=use_IN
                )
            )

        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_filters = (
            out_filters + in_features if dec_lease == 0 else out_filters * 2
        )

    def forward(self, x, drv_exp=None, return_all=False):
        out = x.pop()
        if return_all:
            out_list = [out]
        for up_block in self.up_blocks:
            out = up_block(out, drv_exp=drv_exp)
            if return_all:
                out_list.append(out)
            skip = x.pop()
            out = torch.cat([out, skip], dim=1)
        if return_all:
            out_list.pop()
            out_list.append(out)
            return out, out_list
        return out


class Hourglass(nn.Module):
    """
    Hourglass architecture.
    """

    def __init__(
        self,
        block_expansion,
        in_features,
        num_blocks=3,
        max_features=256,
        Lwarp=False,
        AdaINc=0,
        dec_lease=0,
        use_IN=False
    ):
        super(Hourglass, self).__init__()
        self.encoder = Encoder(
            block_expansion, in_features, num_blocks, max_features, Lwarp, AdaINc, use_IN
        )
        self.decoder = Decoder(
            block_expansion,
            in_features,
            num_blocks,
            dec_lease,
            max_features,
            Lwarp,
            AdaINc,
            use_IN
        )
        self.out_filters = self.decoder.out_filters

    def forward(self, x, drv_exp=None, return_all=False):
        return self.decoder(self.encoder(x, drv_exp=drv_exp), drv_exp=drv_exp, return_all=return_all)


class LayerNorm2d(nn.Module):
    def __init__(self, n_out, affine=True):
        super(LayerNorm2d, self).__init__()
        self.n_out = n_out
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(n_out, 1, 1))
            self.bias = nn.Parameter(torch.zeros(n_out, 1, 1))

    def forward(self, x):
        normalized_shape = x.size()[1:]
        if self.affine:
            return F.layer_norm(
                x,
                normalized_shape,
                self.weight.expand(normalized_shape),
                self.bias.expand(normalized_shape),
            )

        else:
            return F.layer_norm(x, normalized_shape)


class AntiAliasInterpolation2d(nn.Module):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """

    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-((mgrid - mean) ** 2) / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels
        self.scale = scale

    def forward(self, input):
        if self.scale == 1.0:
            return input

        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = F.interpolate(out, scale_factor=(self.scale, self.scale))

        return out


if __name__ == '__main__':
    model = Hourglass(
            block_expansion=64,
            in_features=3,
            max_features=512,
            num_blocks=3,
            Lwarp=False,
            AdaINc=0,
            dec_lease=0,
        )
    print(model)
    x = torch.zeros((2, 3, 256, 256))
    out, out_list = model(x, return_all=True)
    print(out.shape)
    for t in out_list:
        print(t.shape)
