import re
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

from modules.adain import AdaptiveInstanceNorm


class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc, style_nc):
        super().__init__()

        assert config_text.startswith("spade")
        parsed = re.search("spade(\D+)(\d)x\d(\D*)", config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))
        self.hasid = parsed.group(3) == "id"
        if param_free_norm_type == "instance":
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif "batch" in param_free_norm_type:
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError(
                "%s is not a recognized param-free norm type in SPADE"
                % param_free_norm_type
            )

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.label_nc = label_nc
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw), nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        if self.hasid:
            self.mlp_attention = nn.Sequential(
                nn.Conv2d(norm_nc, 1, kernel_size=ks, padding=pw), nn.Sigmoid(),
            )
            self.adain = AdaptiveInstanceNorm(norm_nc, style_nc)

    def forward(self, x, attr_map, id_emb):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        # segmap = F.interpolate(segmap, size=x.size()[2:], mode='bilinear')
        actv = self.mlp_shared(attr_map)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        spade_out = normalized * (1 + gamma) + beta
        if self.hasid:
            attention = self.mlp_attention(x)
            adain_out = self.adain(x, id_emb)

            out = attention * spade_out + (1 - attention) * adain_out
        else:
            out = spade_out
        return out


class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, semantic_nc, style_nc, norm_G):
        super().__init__()
        # Attributes
        self.learned_shortcut = fin != fout
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if "spectral" in norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = norm_G.replace("spectral", "")
        self.norm_0 = SPADE(spade_config_str, fin, semantic_nc, style_nc)
        self.norm_1 = SPADE(spade_config_str, fmiddle, semantic_nc, style_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, semantic_nc, style_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg, class_emb):
        x_s = self.shortcut(x, seg, class_emb)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg, class_emb)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg, class_emb)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg, class_emb):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg, class_emb))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1, inplace=True)


class SPADEGenerator(nn.Module):
    def __init__(
        self,
        label_nc=256,
        class_dim=256,
        conv_dim=64,
        norm_G="spectralspadebatch3x3",
    ):
        super().__init__()

        nf = conv_dim
        self.nf = conv_dim
        self.norm_G = norm_G

        self.conv1 = spectral_norm(nn.ConvTranspose2d(class_dim, nf * 16, 4)) if "spectral" in norm_G else nn.ConvTranspose2d(class_dim, nf * 16, 4)

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, label_nc[0], class_dim, norm_G)

        self.G_middle_0 = SPADEResnetBlock(
            16 * nf, 16 * nf, label_nc[1], class_dim, norm_G
        )
        self.G_middle_1 = SPADEResnetBlock(
            16 * nf, 16 * nf, label_nc[2], class_dim, norm_G
        )

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, label_nc[3], class_dim, norm_G)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, label_nc[4], class_dim, norm_G)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, label_nc[5], class_dim, norm_G)

        final_nc = nf
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, label_nc[6], class_dim, norm_G)

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

    def forward(self, attr_pyramid, class_emb=None):
        if class_emb is None:
            x = torch.randn(
                (attr_pyramid[0].size(0), 256, 1, 1), device=attr_pyramid[0].device
            )
        else:
            x = class_emb.view(class_emb.size(0), class_emb.size(1), 1, 1)
        x = self.conv1(x)
        style4 = F.interpolate(attr_pyramid[0], size=x.shape[2:], mode="bilinear")
        x = self.head_0(x, style4, class_emb)
        x = F.interpolate(x, scale_factor=2, mode="bilinear")

        style8 = F.interpolate(attr_pyramid[0], size=x.shape[2:], mode="bilinear")
        x = self.G_middle_0(x, style8, class_emb)
        x = F.interpolate(x, scale_factor=2, mode="bilinear")

        style16 = F.interpolate(attr_pyramid[0], size=x.shape[2:], mode="bilinear")
        x = self.G_middle_1(x, style16, class_emb)
        x = F.interpolate(x, scale_factor=2, mode="bilinear")

        style32 = F.interpolate(attr_pyramid[0], size=x.shape[2:], mode="bilinear")
        x = self.up_0(x, style32, class_emb)
        x = F.interpolate(x, scale_factor=2, mode="bilinear")

        style64 = F.interpolate(attr_pyramid[1], size=x.shape[2:], mode="bilinear")
        x = self.up_1(x, style64, class_emb)
        x = F.interpolate(x, scale_factor=2, mode="bilinear")

        style128 = F.interpolate(attr_pyramid[2], size=x.shape[2:], mode="bilinear")
        x = self.up_2(x, style128, class_emb)
        x = F.interpolate(x, scale_factor=2, mode="bilinear")

        style256 = F.interpolate(attr_pyramid[3], size=x.shape[2:], mode="bilinear")
        x = self.up_3(x, style256, class_emb)

        x = F.leaky_relu(x, 2e-1, inplace=True)

        x = self.conv_img(x)
        x = torch.tanh(x)

        return x
