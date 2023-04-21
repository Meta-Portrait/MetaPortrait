import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from modules.util import AntiAliasInterpolation2d


class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss. See Sec 3.3.
    """
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class ImagePyramide(torch.nn.Module):
    """
    Create image pyramide for computing pyramide perceptual loss. See Sec 3.3
    """
    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict


def detach_kp(kp):
    return {key: value.detach() for key, value in kp.items()}


class GeneratorFullModel(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator, train_params, arch=None, rank=0, conf=None):
        super(GeneratorFullModel, self).__init__()
        self.arch = arch
        self.conf = conf
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = train_params['scales']
        if conf['model']['discriminator'].get('type', 'MultiPatchGan') == 'MultiPatchGan':
            self.disc_scales = self.discriminator.scales
        
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.to(rank)

        self.loss_weights = train_params['loss_weights']

        if sum(self.loss_weights['perceptual']) != 0:
            self.vgg = Vgg19()
            if torch.cuda.is_available():
                self.vgg = self.vgg.to(rank)
            self.vgg.eval()

        if self.loss_weights.get('warp_ce', 0) > 0:
            self.ce_loss = nn.CrossEntropyLoss().to(rank)
        
        if self.loss_weights.get('l1', 0) > 0:
            self.l1_loss = nn.L1Loss()

    def nist_prec(self, x):
        x = (x.clone() - 0.5) * 2 # -1 ~ 1
        x = x[:, :, 25:256, 25:256]
        x = torch.flip(x,[1]) # RGB -> BGR
        return x
    
    def forward_warp(self, x, cal_loss=True):
        if self.conf['dataset'].get('ldmkimg', False):
            ldmk_line = torch.cat((x['source_line'], x['driving_line']), dim=1)
        else:
            ldmk_line = None
        generated = self.generator(x['source'], ldmk_line=ldmk_line, stage='Warp')

        loss_values = {}
        if cal_loss:
            pyramide_real = self.pyramid(x['driving'])
            pyramide_generated = self.pyramid(generated['deformed'])
            value_total = 0
            for scale in self.scales:
                x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += self.loss_weights['perceptual'][i] * value

            loss_values['warp_perceptual'] = value_total

        return loss_values, generated

    def forward_refine(self, x, warp_out, loss_values, inference=False):
        kp_driving = {'value': x['driving_ldmk']}
        embed_id = x['source_id']
   
        ldmk_line = torch.cat((x['source_line'], x['driving_line']), dim=1) if self.conf['dataset'].get('ldmkimg', False) else None
        if self.loss_weights.get('mouth_enhance', 0) > 0:
            mouth_line = x['driving_line'] * x['mouth_mask']
        else:
            mouth_line = None

        generated = self.generator(x['source'], kp_driving=kp_driving, src_id=embed_id, ldmk_line=ldmk_line, mouth_line=mouth_line, warp_out=warp_out, stage='Refine')
        if inference:
            return generated

        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(generated['prediction'])

        if sum(self.loss_weights['perceptual']) != 0:
            value_total = 0
            eye_total = 0
            mouth_total = 0
            for scale in self.scales:
                x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += self.loss_weights['perceptual'][i] * value

                if self.loss_weights.get('eye_enhance', 0) > 0:
                    eye_scale = F.interpolate(x['eye_mask'], size=pyramide_generated['prediction_' + str(scale)].shape[2:], mode='nearest')
                    eye_total += ((pyramide_generated['prediction_' + str(scale)] - pyramide_real['prediction_' + str(scale)]) ** 2 * eye_scale).sum() / (eye_scale.sum() + 1e-6)

                if self.loss_weights.get('mouth_enhance', 0) > 0:
                    mouth_scale = F.interpolate(x['mouth_mask'], size=pyramide_generated['prediction_' + str(scale)].shape[2:], mode='nearest')
                    mouth_total += ((pyramide_generated['prediction_' + str(scale)] - pyramide_real['prediction_' + str(scale)]) ** 2 * mouth_scale).sum() / (mouth_scale.sum() + 1e-6)

            loss_values['perceptual'] = value_total
            if self.loss_weights.get('eye_enhance', 0) > 0:
                loss_values['eye'] = eye_total * self.loss_weights['eye_enhance']
            if self.loss_weights.get('mouth_enhance', 0) > 0:
                loss_values['mouth'] = mouth_total * self.loss_weights['mouth_enhance']
        
        if self.loss_weights.get('l1', 0) > 0:
            loss_values['l1'] = self.l1_loss(generated['prediction'], x['driving']) * self.loss_weights['l1']

        # if self.loss_weights.get('id', 0) > 0:
        #     gen_grid = F.affine_grid(x['driving_theta'], [x['driving_theta'].shape[0], 3, 256,256], align_corners=True)
        #     gen_nist = F.grid_sample(F.interpolate(generated['prediction'], (256, 256), mode='bilinear'), gen_grid, align_corners=True)

        #     gen_id = self.id_classifier(self.nist_prec(gen_nist))
        #     gen_id = F.normalize(gen_id, dim=1)
        #     tgt_id = F.normalize(embed_id, dim=1)

        #     loss_values['id'] = (1 - (gen_id * tgt_id).sum(1).mean()) * self.loss_weights['id']

        if self.loss_weights['generator_gan'] != 0:
            if self.conf['model']['discriminator'].get('type', 'MultiPatchGan') == 'MultiPatchGan':
                if self.conf['model']['discriminator'].get('use_kp', False):
                    discriminator_maps_generated = self.discriminator(pyramide_generated, kp=detach_kp(kp_driving))
                    discriminator_maps_real = self.discriminator(pyramide_real, kp=detach_kp(kp_driving))
                else:
                    discriminator_maps_generated = self.discriminator(pyramide_generated, kp=x['driving_line'])
                    discriminator_maps_real = self.discriminator(pyramide_real, kp=x['driving_line'])
                value_total = 0
                for scale in self.disc_scales:
                    key = 'prediction_map_%s' % scale
                    value = ((1 - discriminator_maps_generated[key]) ** 2).mean()
                    value_total += self.loss_weights['generator_gan'] * value
                loss_values['gen_gan'] = value_total

                if sum(self.loss_weights['feature_matching']) != 0:
                    value_total = 0
                    for scale in self.disc_scales:
                        key = 'feature_maps_%s' % scale
                        for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_generated[key])):
                            if self.loss_weights['feature_matching'][i] == 0:
                                continue
                            value = torch.abs(a - b).mean()
                            value_total += self.loss_weights['feature_matching'][i] * value
                        loss_values['feature_matching'] = value_total

            else:
                discriminator_maps_generated = self.discriminator(pyramide_generated['prediction_1'])
                value = ((1 - discriminator_maps_generated) ** 2).mean()
                loss_values['gen_gan'] = self.loss_weights['generator_gan'] * value

        return loss_values, generated


    def forward(self, x, stage=None, inference=False):
        if stage == 'Warp':
            return self.forward_warp(x, cal_loss=not inference)
        elif stage == 'Full':
            warp_loss, warp_out = self.forward_warp(x)
            return self.forward_refine(x, warp_out, warp_loss, inference=inference)
        else:
            raise Exception("Unknown stage.")

    def get_gaze_loss(self, deformation, gaze):
        mask = (gaze != 0).detach().float()
        up_deform = F.interpolate(deformation.permute(0,3,1,2), size=gaze.shape[1:3], mode='bilinear').permute(0,2,3,1)
        gaze_loss = (torch.abs(up_deform - gaze) * mask).sum() / (mask.sum() + 1e-6)
        return gaze_loss

    def get_ldmk_loss(self, mask, ldmk_gt):
        pred = F.interpolate(mask, size=ldmk_gt.shape[1:], mode='bilinear')
        ldmk_loss = F.cross_entropy(pred, ldmk_gt, ignore_index=0)
        return ldmk_loss


class DiscriminatorFullModel(torch.nn.Module):
    """
    Merge all discriminator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator, train_params):
        super(DiscriminatorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = self.discriminator.scales
        self.use_kp = discriminator.use_kp
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']

    def forward(self, x, generated):
        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(generated['prediction'].detach())

        if self.use_kp:
            kp_driving = generated['kp_driving']
            discriminator_maps_generated = self.discriminator(pyramide_generated, kp=detach_kp(kp_driving))
            discriminator_maps_real = self.discriminator(pyramide_real, kp=detach_kp(kp_driving))
        else:
            kp_driving = x['driving_line']
            discriminator_maps_generated = self.discriminator(pyramide_generated, kp=kp_driving)
            discriminator_maps_real = self.discriminator(pyramide_real, kp=kp_driving)

        loss_values = {}
        value_total = 0
        for scale in self.scales:
            key = 'prediction_map_%s' % scale
            value = (1 - discriminator_maps_real[key]) ** 2 + discriminator_maps_generated[key] ** 2
            value_total += self.loss_weights['discriminator_gan'] * value.mean()
        loss_values['disc_gan'] = value_total

        if self.loss_weights.get('D_exp', 0) > 0:
            loss_values['exp'] = F.mse_loss(discriminator_maps_real['exp_1'], x['driving_exp']) * self.loss_weights['D_exp'] + \
                F.mse_loss(discriminator_maps_generated['exp_1'], x['driving_exp']) * self.loss_weights['D_exp']

        return loss_values

