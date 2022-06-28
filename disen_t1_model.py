import pdb

import torch
# import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
from torch.nn.modules.utils import _pair
from torch.nn.modules.conv import _ConvNd
from collections import namedtuple
from utils import mkdir_p, create_one_hot, softmax, KLDivergenceLoss, TemperatureAnneal, ShapeLoss, SmoothLoss, NCCLoss
from torch.autograd import Variable
import csv
from copy import deepcopy
from tqdm import tqdm
import os
import torchvision.utils as torch_utils
import nibabel as nib

import functools
import numpy as np
# import resnet

class ChannelAttentionLayer(nn.Module):
    # CVPR2018 squeeze and excitation
    def __init__(self, in_num_ch, sample_factor=16):
        super(ChannelAttentionLayer, self).__init__()

        self.W_down = nn.Linear(in_num_ch, in_num_ch//sample_factor)
        self.W_up = nn.Linear(in_num_ch//sample_factor, in_num_ch)

    def forward(self, x):
        x_gp = torch.mean(x, (2,3))

        x_down = F.relu(self.W_down(x_gp))
        alpha = F.sigmoid(self.W_up(x_down))

        alpha_exp = alpha.unsqueeze(2).unsqueeze(3).expand_as(x)
        out = (1 + alpha_exp) * x
        return out, alpha

class MultiAttentionLayer(nn.Module):
    def __init__(self, in_num_ch, gate_num_ch, sample_factor_spatial=(2,2), sample_factor_channel=16, kernel_stride_ratio=4, is_bn=True):
        super(MultiAttentionLayer, self).__init__()
        self.W_x = nn.Conv2d(in_num_ch, in_num_ch, 1, 1)
        self.W_g = nn.Conv2d(gate_num_ch, in_num_ch, 1, 1)
        self.AvgPool = nn.AvgPool2d(kernel_size=tuple([z * kernel_stride_ratio for z in sample_factor_spatial]), stride=sample_factor_spatial)
        self.W_down = nn.Conv2d(in_num_ch, in_num_ch/sample_factor_channel, 1, 1)
        self.W_up = nn.Conv2d(in_num_ch/sample_factor_channel, in_num_ch, 1, 1)
        if is_bn:
            self.W_out = nn.Sequential(
                nn.Conv2d(in_num_ch, in_num_ch, 1, 1),
                nn.BatchNorm2d(in_num_ch)
            )
        else:
            self.W_out = nn.Conv2d(in_num_ch, in_num_ch, 1, 1)

    def forward(self, x, g):
        # add symmetry, combine x and g_diff
        # pdb.set_trace()
        x_size = x.size()
        x_post = self.W_x(x)
        g_diff = g - torch.flip(g, dims=[2])
        g_post = F.interpolate(self.W_g(g_diff), size=x_size[2:], mode='bilinear')
        xg_post = F.relu(x_post + g_post, inplace=True)

        # channel-wise attention for each spatial sample square
        xg_post_avg = self.AvgPool(xg_post)
        xg_down = F.relu(self.W_down(xg_post_avg))
        alpha = F.sigmoid(self.W_up(xg_down))
        alpha_upsample = F.upsample(alpha, size=x_size[2:], mode='bilinear')

        out = self.W_out((1+alpha_upsample) * x)
        return out, alpha_upsample


class DisQ(nn.Module):
    def __init__(self, beta_dim, theta_dim,
                 train_sample='st_gumbel_softmax', valid_sample='argmax',
                 pretrained_model=None,
                 initial_temp=1.0, anneal_rate=5e-4,
                 device=None, fine_tune=False):
        super(DisQ, self).__init__()
        self.beta_dim = beta_dim
        self.theta_dim = theta_dim
        self.train_sample = train_sample
        self.valid_sample = valid_sample
        self.initial_temp = initial_temp
        self.anneal_rate = anneal_rate
        self.device = device
        self.fine_tune = fine_tune if pretrained_model is not None else False

        if self.fine_tune:
            print('Fine tuning network...')

        # define networks
        self.beta_encoder = Unet(in_ch=1, out_ch=16, num_lvs=4, base_ch=8, final_act='leakyrelu')
        self.da_beta = DomainAdaptorBeta(in_ch=16, out_ch=self.beta_dim, final_act=False)
        self.theta_encoder = ThetaEncoder(in_ch=1, out_ch=128)
        self.da_theta = DomainAdaptorTheta(out_ch=self.theta_dim)
        self.af_theta = AffineTheta(out_ch=self.theta_dim)
        self.decoder = Unet(in_ch=self.theta_dim+self.beta_dim, num_lvs=4, base_ch=16, out_ch=1, final_act='leakyrelu')
        #self.decoder_affine = Unet(in_ch=self.theta_dim+self.beta_dim+self.theta_dim*self.beta_dim, num_lvs=4, base_ch=16, out_ch=1, final_act='leakyrelu')
        self.decoder_affine = Unet(in_ch=self.theta_dim+self.beta_dim+self.theta_dim, num_lvs=4, base_ch=16, out_ch=1, final_act='leakyrelu')
        self.vgg = Vgg16(requires_grad=False)

        # initialize training variables
        self.train_loader, self.valid_loader = None, None
        self.out_dir = None
        self.batch_size = None
        self.optim_beta_encoder, self.optim_theta_encoder, self.optim_decoder = None, None, None
        self.optim_da_beta, self.optim_da_theta = None, None
        self.temp_sched = None

        # pretrained models
        self.checkpoint = None
        if pretrained_model is not None:
            self.checkpoint = torch.load(pretrained_model, map_location=self.device)
            self.beta_encoder.load_state_dict(self.checkpoint['beta_encoder'])
            self.theta_encoder.load_state_dict(self.checkpoint['theta_encoder'])
            self.decoder.load_state_dict(self.checkpoint['decoder'])
            self.da_beta.load_state_dict(self.checkpoint['da_beta'])
            self.da_theta.load_state_dict(self.checkpoint['da_theta'])

        # send to device
        self.beta_encoder.to(self.device)
        self.theta_encoder.to(self.device)
        self.decoder.to(self.device)
        self.decoder_affine.to(self.device)
        self.vgg.to(self.device)
        self.da_beta.to(self.device)
        self.da_theta.to(self.device)
        self.af_theta.to(self.device)
        self.start_epoch = 0

    def initialize_training(self, out_dir, lr):

        self.kld_loss = KLDivergenceLoss(reduction='none')
        self.l1_loss = nn.L1Loss(reduction='none')
        self.l2_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none', pos_weight=10.0*torch.ones(1,).to(self.device))
        self.shape_loss = ShapeLoss(reduction='none')

    def reparameterize_logit(self, logit, method, temp_sched):
        tau = temp_sched.get_temp() if temp_sched is not None else 0.5
        if method == 'gumbel_softmax':
            beta = F.gumbel_softmax(logit, tau=tau, dim=1, hard=False)
        elif method == 'st_gumbel_softmax':
            beta = F.gumbel_softmax(logit, tau=tau, dim=1, hard=True)
        elif method == 'argmax':
            beta = create_one_hot(logit, dim=1)
        else:
            beta = softmax(logit, temp_sched.get_temp(), dim=1)
        return beta

    def cal_beta(self, imgs, method, temp_sched):
        logits = []
        betas = []
        for img in imgs:
            logit = self.beta_encoder(img)
            logit = self.da_beta(logit)
            beta = self.reparameterize_logit(logit, method, temp_sched)
            logits.append(logit)
            betas.append(beta)
        return tuple(betas), tuple(logits)

    def cal_theta(self, imgs):
        thetas = []
        mus = []
        logvars = []
        for img in imgs:
            theta, mu, logvar = self.da_theta(self.theta_encoder(img), self.device)
            thetas.append(theta)
            mus.append(mu)
            logvars.append(logvar)
        return thetas, mus, logvars

    def affine_theta(self, thetas):
        amus = []
        alogvars = []
        for theta in thetas:
            amu, alogvar = self.af_theta(theta, self.device)
            amus.append(amu)
            alogvars.append(alogvar)
        return amus, alogvars

    def anatomy_transformation(self, amus, alogvars, betas):
        affine_zs = []
        for i in range(len(betas)):
            beta = betas[i]
            mu = amus[i]
            logvar = alogvars[i]
            tmp_z = []
            for j in range(logvar.size(1)):
                affine_z = torch.exp(logvar[:, j:j+1, :, :]) * beta + mu[:, j:j+1, :, :]
                tmp_z.append(affine_z)
            affine_z = torch.cat(tmp_z, 1)
            affine_zs.append(affine_z)
        return affine_zs

    def cat_betaz(self, betas, zs):
        ress = []
        for i in range(len(betas)):
            beta = betas[i]
            z = zs[i]
            res = torch.cat([beta, z], 1)
            ress.append(res)
        return ress

    def cat_betamu(self, betas, zs):
        ress = []
        for i in range(len(betas)):
            beta = betas[i]
            z = zs[i]
            res = torch.cat([beta, z.repeat(1,1,beta.shape[2],beta.shape[3])], 1)
            ress.append(res)
        return ress

    def decode_aff(self, betas, thetas, swap_beta=True, self_rec=False):
        rec_imgs = []
        img_ids = []  # which modality is used during decoding
        beta_combined = torch.cat(betas, dim=1)
        num_modalities = len(betas)
        for img_id, theta in enumerate(thetas):
            if swap_beta:
                beta_id = [np.random.randint(num_modalities) * self.beta_dim + i for i in range(self.beta_dim)]
                beta = beta_combined[:, beta_id, :, :]
                combined_img = torch.cat([beta, theta.repeat(1,1,beta.shape[2],beta.shape[3])], dim=1)
                rec_img = self.decoder(combined_img)
                rec_imgs.append(rec_img)
                img_ids.append(img_id)
            else:
                for modality_id in range(num_modalities):
                    # dim_num = self.beta_dim + self.beta_dim * self.theta_dim
                    dim_num = self.beta_dim + self.theta_dim
                    beta_id = [modality_id * dim_num + i for i in range(dim_num)]
                    beta = beta_combined[:, beta_id, :, :]
                    combined_img = torch.cat([beta, theta.repeat(1,1,beta.shape[2],beta.shape[3])], dim=1)
                    rec_img = self.decoder_affine(combined_img)
                    if self_rec:
                        img_ids.append(img_id)
                        rec_imgs.append(rec_img)
                    else:
                        if modality_id != img_id:
                            img_ids.append(img_id)
                            rec_imgs.append(rec_img)
        return tuple(rec_imgs), tuple(img_ids)

class Transformer_2D(nn.Module):
    def __init__(self):
        super(Transformer_2D, self).__init__()
    # @staticmethod
    def forward(self, src, flow):
        b = flow.shape[0]
        h = flow.shape[2]
        w = flow.shape[3]

        size = (h,w)

        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = grid.to(torch.float32)
        grid = grid.repeat(b,1,1,1).cuda()
        new_locs = grid+flow
        shape = flow.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 1)
        new_locs = new_locs[..., [1 , 0]]
        warped = F.grid_sample(src,new_locs,align_corners=True)
        # ctx.save_for_backward(src,flow)
        return warped


class Reg(nn.Module):
    def __init__(self, h, w, src_feats=1, trg_feats=1, device=None):
        super(Reg, self).__init__()
        self.init_func = 'kaiming'
        self.init_to_identity = True

        self.device = device

        self.h, self.w = h, w
        self.in_cha = src_feats
        self.in_chb = trg_feats

        # self.num_df = [32, 64, 64, 64, 64, 64, 64]
        # self.num_uf = [64, 64, 64, 64, 64, 64, 32]

        self.num_df = [32, 64, 64, 64, 64, 64]
        self.num_uf = [64, 64, 64, 64, 64, 32]

        self.offset_map = ResUnet(self.in_cha, self.in_chb, self.num_df, self.num_uf, init_func=self.init_func, init_to_identity=self.init_to_identity)
        self.spatial_transform = Transformer_2D()

        self.identity_grid = self.get_identity_grid()

        self.offset_map.to(self.device)
        self.spatial_transform.to(self.device)

    def initialize_training(self):
        self.corr_loss = nn.L1Loss(reduction='none')
        self.smooth_loss = SmoothLoss(reduction='none')
        self.ncc_loss = NCCLoss()

    def get_identity_grid(self):
        x = torch.linspace(-1.0, 1.0, self.w)
        y = torch.linspace(-1.0, 1.0, self.h)
        xx, yy = torch.meshgrid([y, x])
        xx = xx.unsqueeze(dim=0)
        yy = yy.unsqueeze(dim=0)
        identity = torch.cat((yy, xx), dim=0).unsqueeze(0)
        return identity

    def forward(self, img_a, img_b, apply_on=None):

        deformations = self.offset_map(img_a, img_b)

        return deformations


class ResUnet(torch.nn.Module):
    def __init__(self, nc_a, nc_b, num_df, num_uf, init_func, init_to_identity, use_down_resblocks=True, resnet_nblocks=3, refine_output=True):
        super(ResUnet, self).__init__()
        self.act = 'leaky_relu'
        self.ndown_blocks = len(num_df)
        self.nup_blocks = len(num_uf)
        assert self.ndown_blocks >= self.nup_blocks
        in_nf = nc_a + nc_b

        conv_num = 1
        skip_nf = {}
        for out_nf in num_df:
            setattr(self, 'down_{}'.format(conv_num),
                    ResDownBlock(in_nf, out_nf, 3, 1, 1, activation=self.act, init_func=init_func, bias=True,
                              use_resnet=use_down_resblocks, use_norm=False))
            skip_nf['down_{}'.format(conv_num)] = out_nf
            in_nf = out_nf
            conv_num += 1
        conv_num -= 1
        if use_down_resblocks:
            self.c1 = ResConv(in_nf, 2 * in_nf, 1, 1, 0, activation=self.act, init_func=init_func, bias=True,
                           use_resnet=False, use_norm=False)
            self.t = ((lambda x: x) if resnet_nblocks == 0
                      else ResnetTransformer(2 * in_nf, resnet_nblocks, init_func))
            self.c2 = ResConv(2 * in_nf, in_nf, 1, 1, 0, activation=self.act, init_func=init_func, bias=True,
                           use_resnet=False, use_norm=False)
        # ------------- Up-sampling path

        for out_nf in num_uf:
            setattr(self, 'up_{}'.format(conv_num),
                    ResConv(in_nf + skip_nf['down_{}'.format(conv_num)], out_nf, 3, 1, 1, bias=True, activation=self.act,
                         init_fun=init_func, use_norm=False, use_resnet=False))
            in_nf = out_nf
            conv_num -= 1
        if refine_output:
            self.refine = nn.Sequential(ResnetTransformer(in_nf, 1, init_func),
                                        ResConv(in_nf, in_nf, 1, 1, 0, use_resnet=False, init_func=init_func,
                                             activation=self.act,
                                             use_norm=False)
                                        )
        else:
            self.refine = lambda x: x
        self.output = ResConv(in_nf, 2, 3, 1, 1, use_resnet=False, bias=True,
                           init_func=('zeros' if init_to_identity else init_func), activation=None,
                           use_norm=False)

    def forward(self, img_a, img_b):
        x = torch.cat([img_a, img_b], 1)
        skip_vals = {}
        conv_num = 1
        # Down
        while conv_num <= self.ndown_blocks:
            x, skip = getattr(self, 'down_{}'.format(conv_num))(x)
            skip_vals['down_{}'.format(conv_num)] = skip
            conv_num += 1
        if hasattr(self, 't'):
            x = self.c1(x)
            x = self.t(x)
            x = self.c2(x)
        # Up
        conv_num -= 1
        while conv_num > (self.ndown_blocks - self.nup_blocks):
            s = skip_vals['down_{}'.format(conv_num)]
            x = F.interpolate(x, (s.size(2), s.size(3)), mode='bilinear')
            x = torch.cat([x, s], 1)
            x = getattr(self, 'up_{}'.format(conv_num))(x)
            conv_num -= 1
        x = self.refine(x)
        x = self.output(x)
        return x


class ResDownBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False, activation='relu',
                 init_func='kaiming', use_norm=False, use_resnet=False, skip=True, refine=False, pool=True,
                 pool_size=2, **kwargs):
        super(ResDownBlock, self).__init__()
        self.conv_0 = ResConv(in_channels, out_channels, kernel_size, stride, padding, bias=bias,
                           activation=activation, init_func=init_func, use_norm=use_norm, callback=None,
                           use_resnet=use_resnet, **kwargs)
        self.conv_1 = None
        if refine:
            self.conv_1 = ResConv(out_channels, out_channels, kernel_size, stride, padding, bias=bias,
                               activation=activation, init_func=init_func, use_norm=use_norm, callback=None,
                               use_resnet=use_resnet, **kwargs)
        self.skip = skip
        self.pool = None
        if pool:
            self.pool = nn.MaxPool2d(kernel_size=pool_size)

    def forward(self, x):
        x = skip = self.conv_0(x)
        if self.conv_1 is not None:
            x = skip = self.conv_1(x)
        if self.pool is not None:
            x = self.pool(x)
        if self.skip:
            return x, skip
        else:
            return x

resnet_n_blocks = 1
from functools import partial
norm_layer = partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)

class ResConv(torch.nn.Module):
    """Defines a basic convolution layer.
    The general structure is as follow:

    Conv -> Norm (optional) -> Activation -----------> + --> Output
                                         |            ^
                                         |__ResBlcok__| (optional)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, activation='relu',
                 init_func='kaiming', use_norm=False, use_resnet=False, **kwargs):
        super(ResConv, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.resnet_block = ResnetTransformer(out_channels, resnet_n_blocks, init_func) if use_resnet else None
        self.norm = norm_layer(out_channels) if use_norm else None
        self.activation = get_activation(activation, **kwargs)
        # Initialize the weights
        init_ = get_init_function(activation, init_func)
        init_(self.conv2d.weight)
        if self.conv2d.bias is not None:
            self.conv2d.bias.data.zero_()
        if self.norm is not None and isinstance(self.norm, nn.BatchNorm2d):
            nn.init.normal_(self.norm.weight.data, 0.0, 1.0)
            nn.init.constant_(self.norm.bias.data, 0.0)

    def forward(self, x):
        x = self.conv2d(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.resnet_block is not None:
            x = self.resnet_block(x)
        return x

class ResnetTransformer(torch.nn.Module):
    def __init__(self, dim, n_blocks, init_func):
        super(ResnetTransformer, self).__init__()
        model = []
        for i in range(n_blocks):  # add ResNet blocks
            model += [
                ResnetBlock(dim, padding_type='reflect', norm_layer=norm_layer, use_dropout=False,
                            use_bias=True)]
        self.model = nn.Sequential(*model)

        init_ = get_init_function('relu', init_func)

        def init_weights(m):
            if type(m) == nn.Conv2d:
                init_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            if type(m) == nn.BatchNorm2d:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

        self.model.apply(init_weights)

    def forward(self, x):
        return self.model(x)

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

def get_activation(activation, **kwargs):
    """Get the appropriate activation from the given name"""
    if activation == 'relu':
        return nn.ReLU(inplace=False)
    elif activation == 'leaky_relu':
        negative_slope = 0.2 if 'negative_slope' not in kwargs else kwargs['negative_slope']
        return nn.LeakyReLU(negative_slope=negative_slope, inplace=False)
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    else:
        return None

def get_init_function(activation, init_function, **kwargs):
    """Get the initialization function from the given name."""
    a = 0.0
    if activation == 'leaky_relu':
        a = 0.2 if 'negative_slope' not in kwargs else kwargs['negative_slope']

    gain = 0.02 if 'gain' not in kwargs else kwargs['gain']
    if isinstance(init_function, str):
        if init_function == 'kaiming':
            activation = 'relu' if activation is None else activation
            return partial(torch.nn.init.kaiming_normal_, a=a, nonlinearity=activation, mode='fan_in')
        elif init_function == 'dirac':
            return torch.nn.init.dirac_
        elif init_function == 'xavier':
            activation = 'relu' if activation is None else activation
            gain = torch.nn.init.calculate_gain(nonlinearity=activation, param=a)
            return partial(torch.nn.init.xavier_normal_, gain=gain)
        elif init_function == 'normal':
            return partial(torch.nn.init.normal_, mean=0.0, std=gain)
        elif init_function == 'orthogonal':
            return partial(torch.nn.init.orthogonal_, gain=gain)
        elif init_function == 'zeros':
            return partial(torch.nn.init.normal_, mean=0.0, std=1e-5)
    elif init_function is None:
        if activation in ['relu', 'leaky_relu']:
            return partial(torch.nn.init.kaiming_normal_, a=a, nonlinearity=activation)
        if activation in ['tanh', 'sigmoid']:
            gain = torch.nn.init.calculate_gain(nonlinearity=activation, param=a)
            return partial(torch.nn.init.xavier_normal_, gain=gain)
    else:
        return init_function


class Vgg16(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.module1 = nn.Sequential()
        self.module2 = nn.Sequential()
        self.module3 = nn.Sequential()
        self.module4 = nn.Sequential()

        for x in range(4):
            self.module1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.module2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.module3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.module4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.module1(X)
        h_relu1_2 = h
        h = self.module2(h)
        h_relu2_2 = h
        h = self.module3(h)
        h_relu3_3 = h
        h = self.module4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out

class Unet(nn.Module):
    def __init__(self, in_ch, out_ch, num_lvs=4, base_ch=16, final_act='noact'):
        super(Unet, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.num_lvs = num_lvs
        self.base_ch = base_ch
        self.final_act = final_act

        self.in_conv = nn.Conv2d(in_ch, self.base_ch, 3, 1, 1)

        for lv in range(self.num_lvs):
            channel = self.base_ch * (2 ** lv)
            self.add_module(f'downconv_{lv}', ConvBlock2d(channel, channel*2, channel*2))
            self.add_module(f'maxpool_{lv}', nn.MaxPool2d(kernel_size=2, stride=2))
            self.add_module(f'upsample_{lv}', Upsample(channel*4))
            self.add_module(f'upconv_{lv}', ConvBlock2d(channel*4, channel*2, channel*2))

        bttm_ch = self.base_ch * (2 ** self.num_lvs)
        self.bttm_conv = ConvBlock2d(bttm_ch, bttm_ch*2, bttm_ch*2)

        self.out_conv = nn.Conv2d(self.base_ch*2, self.out_ch, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, in_tensor):
        concat_out = {}
        x = self.in_conv(in_tensor)
        for lv in range(self.num_lvs):
            concat_out[lv] = getattr(self, f'downconv_{lv}')(x)
            x = getattr(self, f'maxpool_{lv}')(concat_out[lv])
        x = self.bttm_conv(x)
        for lv in range(self.num_lvs-1, -1, -1):
            x = getattr(self, f'upsample_{lv}')(x, concat_out[lv])
            x = getattr(self, f'upconv_{lv}')(x)
        x = self.out_conv(x)

        if self.final_act == 'sigmoid':
            return self.sigmoid(x)
        elif self.final_act == 'leakyrelu':
            return self.leakyrelu(x)
        elif self.final_act == 'relu':
            return self.relu(x)
        else:
            return x

class ConvBlock2d(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(ConvBlock2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, 1, 1),
            nn.InstanceNorm2d(mid_ch, affine=True),
            nn.LeakyReLU(),
            nn.Conv2d(mid_ch, out_ch, 3, 1, 1),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU())

    def forward(self, in_tensor):
        return self.conv(in_tensor)

class Upsample(nn.Module):
    def __init__(self, in_ch):
        super(Upsample, self).__init__()
        self.out_ch = int(in_ch / 2)
        self.conv = nn.Conv2d(in_ch, self.out_ch, 3, 1, 1)
        self.norm = nn.InstanceNorm2d(self.out_ch, affine=True)
        self.act = nn.LeakyReLU()

    def forward(self, in_tensor, ori):
        upsmp = F.interpolate(in_tensor, size=None, scale_factor=2, mode='bilinear', align_corners=True)
        upsmp = self.act(self.norm(self.conv(upsmp)))
        output = torch.cat((ori, upsmp), dim=1)
        return output

class ThetaEncoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ThetaEncoder, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.conv = nn.Sequential(
            nn.Conv2d(self.in_ch, 32, 4, 2, 1),
            nn.InstanceNorm2d(32, affine=True),
            nn.LeakyReLU(), # 256*256 --> 128*128
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(), # 128*128 --> 64*64
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(), # 64*64--> 32*32
            nn.Conv2d(128, self.out_ch, 4, 2, 1),
            nn.InstanceNorm2d(self.out_ch, affine=True),
            nn.LeakyReLU()) # 32*32--> 16*16

    def forward(self, in_tensor):
        return self.conv(in_tensor)

class DomainAdaptorBeta(nn.Module):
    def __init__(self, in_ch, out_ch, final_act=True):
        super(DomainAdaptorBeta, self).__init__()
        self.final_act = final_act
        self.da = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, 1, 1),
            nn.InstanceNorm2d(16, affine=True),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.InstanceNorm2d(32, affine=True),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(),
            nn.Conv2d(64, out_ch, 3, 1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, in_tensor):
        if self.final_act:
            return self.sigmoid(self.da(in_tensor))
        else:
            return self.da(in_tensor)

class DomainAdaptorTheta(nn.Module):
    def __init__(self, out_ch):
        super(DomainAdaptorTheta, self).__init__()
        self.out_ch = out_ch
        self.mean_conv = nn.Sequential(
            nn.Conv2d(128, 32, 1, 1, 0),
            nn.InstanceNorm2d(32, affine=True),
            nn.LeakyReLU(),
            nn.Conv2d(32, self.out_ch, 12, 12, 0))  # gendata: 12*12, mapping: 16*16
        self.logvar_conv = nn.Sequential(
            nn.Conv2d(128, 32, 1, 1, 0),
            nn.InstanceNorm2d(32, affine=True),
            nn.LeakyReLU(),
            nn.Conv2d(32, self.out_ch, 12, 12, 0))

    def forward(self, in_tensor, device):
        mu = self.mean_conv(in_tensor)
        logvar = self.logvar_conv(in_tensor)
        theta = self.sample(mu, logvar, device)
        #print(theta.shape)
        return theta, mu, logvar

    def sample(self, mu, logvar, device):
        theta = torch.randn(mu.size()).to(device) * torch.sqrt(torch.exp(logvar)) + mu
        return theta

class AffineTheta(nn.Module):
    def __init__(self, out_ch):
        super(AffineTheta, self).__init__()
        self.in_ch = out_ch
        self.out_ch = out_ch
        self.mean_fc = nn.Sequential(
            nn.Linear(self.in_ch, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 20),
            nn.LeakyReLU(),
            nn.Linear(20, self.out_ch))
        self.logvar_fc = nn.Sequential(
            nn.Linear(self.in_ch, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 20),
            nn.LeakyReLU(),
            nn.Linear(20, self.out_ch))

    def forward(self, in_tensor, device):
        x = in_tensor.view(in_tensor.size(0), -1)
        mu = self.mean_fc(x)
        mu = mu.view(in_tensor.size(0), in_tensor.size(1), in_tensor.size(2), -1)
        logvar = self.logvar_fc(x)
        logvar = logvar.view(in_tensor.size(0), in_tensor.size(1), in_tensor.size(2), -1)
        #print(theta.shape)
        return mu, logvar

    def sample(self, mu, logvar, device):
        theta = torch.randn(mu.size()).to(device) * torch.sqrt(torch.exp(logvar)) + mu
        return theta

