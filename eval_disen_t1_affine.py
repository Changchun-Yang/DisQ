#!/usr/bin/env python

import argparse
import os.path
import time
import generators
from losses import *
import scipy.stats as st
from utils import *
from visualizer import Visualizer
import torch.backends.cudnn as cudnn
import logging.config
from neurite import plot
from disen_t1_model import *
import pdb
import scipy.io as scio

# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
parser.add_argument('--data_path', type=str, default='data_t1mapping/', help='the data path')
parser.add_argument('--vis_env', type=str, default='dis_t1', help='visualization environment')
parser.add_argument('--multichannel', default=False, help='specify that data has multiple channels')

# training parameters
parser.add_argument('--gpu', default='0', help='GPU ID number(s), comma-separated (default: 0)')
parser.add_argument('--batch_size', type=int, default=8, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=500,
                    help='number of training epochs (default: 1500)')
parser.add_argument('--steps-per-epoch', type=int, default=100,
                    help='frequency of model saves (default: 100)')
parser.add_argument('--initial-epoch', type=int, default=0,
                    help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=3e-4, help='learning rate (default: 1e-4)')
parser.add_argument('--cudnn-nondet', action='store_true',
                    help='disable cudnn determinism - might slow down training')

# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+',
                    help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec', type=int, nargs='+',
                    help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
parser.add_argument('--int-steps', type=int, default=0,
                    help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=2,
                    help='flow downsample factor for integration (default: 2)')
parser.add_argument('--bidir', default=False, action='store_true', help='enable bidirectional cost function')
parser.add_argument('--lambda_adv_s', default=0.)
parser.add_argument('--is_discrim_s', default=False)
parser.add_argument('--is_distri_z', default=False)
parser.add_argument('--is_cond', default=False)
parser.add_argument('--contrast_list', default=['T1-mapping-M', 'T1-mapping-F'])
parser.add_argument('--block_size', default=1)
parser.add_argument('--in_num_ch', default=1)
parser.add_argument('--out_num_ch', default=1)
#2022_1_30_15_11 [2 2] l1 : BEST

parser.add_argument('--ckpt_timelabel', default='2022_2_22_19_34')
parser.add_argument('--phase', default='train')
parser.add_argument('--continue_train', default=False)
parser.add_argument('--fix_pretrain', default=False)
parser.add_argument('--dataset_name', default='data_t1mapping')
parser.add_argument('--model_name', default='DisT1')
parser.add_argument('--ckpt_path', default='checkpoint')
parser.add_argument('--ckpt_name', default='model_best.pth.tar')
parser.add_argument('--shuffle', default=True)
parser.add_argument('--norm_type', default='z-score')
parser.add_argument('--target_output_act', default='softplus')
parser.add_argument('--input_output_act', default='softplus')
parser.add_argument('--input_height', default=256)
parser.add_argument('--input_width', default=256)
parser.add_argument('--s_num_ch', default=1)
parser.add_argument('--z_size', default=16)
parser.add_argument('--s_compact_method', default='max')
parser.add_argument('--s_sim_method', default='cosine')
parser.add_argument('--z_sim_method', default='cosine')
parser.add_argument('--shared_ana_enc', default=True)
parser.add_argument('--shared_mod_enc', default=True)
parser.add_argument('--shared_inp_dec', default=True)
parser.add_argument('--target_model_name', default='U+SA')
parser.add_argument('--fuse_method', default='mean')
parser.add_argument('--others', default={'mod_enc_s': False, 'ana_dec_act': 'softmax', 'old': False, 'softmax_remove_mask': False})
parser.add_argument('--lambda_recon_x', default=1.0)
parser.add_argument('--lambda_recon_x_mix', default=2.0)
parser.add_argument('--lambda_sim_s', default=10.0)
parser.add_argument('--lambda_sim_z', default=0.0)
# parser.add_argument('--lambda_sim_s', default=0.)
# parser.add_argument('--lambda_sim_z', default=2.0)
parser.add_argument('--lambda_kl', default=0.)
# parser.add_argument('--lambda_latent_z', default=0.1)
parser.add_argument('--lambda_latent_z', default=0.1)
parser.add_argument('--p', default=1)
parser.add_argument('--initial-temp', type=float, default=1.0)
parser.add_argument('--anneal-rate', type=float, default=5e-3)
parser.add_argument('--fine-tune', default=False, action='store_true')


args = parser.parse_args()


def load_mat(test_file, slice_num):
    mat = scio.loadmat(test_file)
    vol = mat['volume']

    vol = np.transpose(vol, (2, 3, 0, 1))

    vol = crop_and_fill(vol, 256)

    sel_vol = vol[1]

    # Get the threshold of 98% pixels of the histogram
    # arr = sel_vol.flatten()
    # n, bins, patches = plt.hist(arr, bins=int(np.max(arr)), cumulative=True)
    # l_d = np.argwhere(n >= arr.shape[0] * 0.98)

    percent_index = np.percentile(sel_vol, 98)
    slices = []
    for i in range(slice_num):
        slice0 = np.clip(sel_vol[i], 0, percent_index - 1) / (percent_index - 1)
        slices.append(slice0[np.newaxis, ...])

    slices = np.concatenate(slices, axis=0)
    return slices, percent_index


def train():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_file = []
    data_path = os.path.join("tmp_eval", "test_ori_data/")

    for f in sorted(os.listdir(data_path)):
        test_file.append(f)

    # test_file.append(os.path.join(args.dataset_name, "Val/9353628_20140129_MOLLI_post_org.mat"))
    # test_file.append(os.path.join(args.dataset_name, "Train/6945955_20140108_MOLLI_post_org.mat"))
    # test_file.append(os.path.join(args.dataset_name, "Test/8185417_20140604_MOLLI_post_org.mat"))

    model = DisQ(beta_dim = 3,
                     theta_dim = 2,
                     train_sample = 'st_gumbel_softmax',
                     valid_sample = 'argmax',
                     pretrained_model = None,
                     initial_temp = args.initial_temp,
                     anneal_rate = args.anneal_rate,
                     device = device,
                     fine_tune = args.fine_tune)

    optim_beta_encoder = torch.optim.Adam(model.beta_encoder.parameters(), lr=args.lr)
    optim_theta_encoder = torch.optim.Adam(model.theta_encoder.parameters(), lr=args.lr)
    optim_decoder = torch.optim.Adam(model.decoder.parameters(), lr=args.lr)
    optim_da_beta = torch.optim.Adam(model.da_beta.parameters(), lr=args.lr)
    optim_da_theta = torch.optim.Adam(model.da_theta.parameters(), lr=args.lr)
    optim_decoder_affine = torch.optim.Adam(model.decoder_affine.parameters(), lr=args.lr)
    optim_af_theta = torch.optim.Adam(model.af_theta.parameters(), lr=args.lr)
    temp_sched = TemperatureAnneal(initial_temp=model.initial_temp, anneal_rate=model.anneal_rate, min_temp=0.5, device=model.device)

    args.ckpt_path = os.path.join(args.ckpt_path, args.dataset_name, args.model_name, args.ckpt_timelabel)

    [model], start_epoch = load_checkpoint_by_key([model], args.ckpt_path, ['model'], device, 'model_best.pth.tar')

    batch_size = 11

    for file_num in range(len(test_file)):
        slices, percent_index = load_mat(os.path.join(data_path, test_file[file_num]), batch_size)

        swap_img = []
        theta_a = []
        theta_b = []
        mu_a = []
        mu_b = []
        amu_a = []
        amu_b = []
        beta_a = []
        beta_b = []

        for i in range(batch_size):
            scan1 = slices[i]
            scan2 = slices[3]
            scan1 = scan1[np.newaxis, ...][np.newaxis, ...]
            scan2 = scan2[np.newaxis, ...][np.newaxis, ...]
            inputs = [scan1, scan2]

            inputs = [torch.from_numpy(d).to(device).float() for d in inputs]

            betas, logits = model.cal_beta(inputs, model.train_sample, temp_sched)
            thetas, mus, logvars = model.cal_theta(inputs)

            amus, alogvars = model.affine_theta(thetas)

            affine_zs = model.anatomy_transformation(amus, alogvars, betas)

            cat_betas = model.cat_betaz(betas, affine_zs)

            rec_imgs, img_ids = model.decode_aff(cat_betas, mus, swap_beta=False, self_rec=True)

            inputs = [d.detach().cpu().numpy() for d in inputs]
            betas = [d.detach().cpu().numpy() for d in betas]
            rec_imgs = [d.detach().cpu().numpy() for d in rec_imgs]
            thetas = [d.detach().cpu().numpy() for d in thetas]
            mus = [d.detach().cpu().numpy() for d in mus]
            amus = [d.detach().cpu().numpy() for d in amus]

            swap_img.append(np.squeeze(rec_imgs[1])[np.newaxis, ...])
            theta_a.append(np.squeeze(thetas[0])[np.newaxis, ...])
            theta_b.append(np.squeeze(thetas[1])[np.newaxis, ...])
            mu_a.append(np.squeeze(mus[0])[np.newaxis, ...])
            mu_b.append(np.squeeze(mus[1])[np.newaxis, ...])
            amu_a.append(np.squeeze(amus[0])[np.newaxis, ...])
            amu_b.append(np.squeeze(amus[1])[np.newaxis, ...])
            beta_a.append(np.squeeze(betas[0])[np.newaxis, ...])
            beta_b.append(np.squeeze(betas[1])[np.newaxis, ...])

        #stretch data
        swap_img = np.concatenate(swap_img, axis=0)

        theta_a = np.concatenate(theta_a, axis=0)
        theta_b = np.concatenate(theta_b, axis=0)
        mu_a = np.concatenate(mu_a, axis=0)
        mu_b = np.concatenate(mu_b, axis=0)
        amu_a = np.concatenate(amu_a, axis=0)
        amu_b = np.concatenate(amu_b, axis=0)
        beta_a = np.concatenate(beta_a, axis=0)
        beta_b = np.concatenate(beta_b, axis=0)

        for i in range(batch_size):
            a = np.min(slices[i])
            b = np.max(slices[i])
            c = np.min(swap_img[i])
            d = np.max(swap_img[i])
            swap_img[i] = [(b - a) / (d - c)] * swap_img[i] + [(a * d - b * c) / (d - c)]    # [c, d] ---> [a, b]
            slices[i] = slices[i] * (percent_index - 1)
            swap_img[i] = swap_img[i] * (percent_index - 1)

        slices = np.transpose(slices, (1, 2, 0))
        swap_img = np.transpose(swap_img, (1, 2, 0))

        mat_set_path = os.path.join("tmp_eval/test_pro_data_%s" % args.ckpt_timelabel, test_file[file_num][:-4]+'_2.mat')
        # scio.savemat(mat_set_path, {'x0': inputs[0][0][0], 'x1': inputs[1][0][0],
        #                             's00': betas[0][0][0], 's10': betas[1][0][0],
        #                             's01': betas[0][0][1], 's11': betas[1][0][1],
        #                             # 's02': betas[0][0][2], 's12': betas[1][0][2],
        #                             'x00': rec_imgs[0][0][0], 'x01': rec_imgs[1][0][0],
        #                             'x10': rec_imgs[2][0][0], 'x11': rec_imgs[3][0][0]})
        # scio.savemat(mat_set_path, {'x03': rec_imgs[1][0][0], 'x30': rec_imgs[2][0][0],
        #                             'x13': rec_imgs[1][1][0], 'x31': rec_imgs[2][1][0],
        #                             'x23': rec_imgs[1][2][0], 'x32': rec_imgs[2][2][0],
        #                             'x33A': rec_imgs[1][3][0], 'x33B': rec_imgs[2][3][0]})
        scio.savemat(mat_set_path, {'ori_data': slices, 'pro_data': swap_img,
                                    'theta_a': theta_a, 'theta_b': theta_b,
                                    'mu_a': mu_a,       'mu_b': mu_b,
                                    'amu_a': amu_a,       'amu_b': amu_b,
                                    'beta_a': beta_a, 'beta_b': beta_b})


if __name__ == '__main__':
    # train()  ## train and eval
    train()   ## test
