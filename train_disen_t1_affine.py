#!/usr/bin/env python

import argparse
import os.path
import time

import torch

import generators
import scipy.stats as st
from utils import *
from visualizer import Visualizer
import torch.backends.cudnn as cudnn
import logging.config
from disen_t1_model import *
import pdb

seed = 10
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic=True

# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
parser.add_argument('--data_path', type=str, default='data_t1mapping/', help='the data path')
parser.add_argument('--vis_env', type=str, default='dis_t1_affine', help='visualization environment')
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
parser.add_argument('--ckpt_timelabel', default='2022_1_24_23_18')
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


if args.ckpt_timelabel and (args.phase == 'test' or args.continue_train is True):
    time_label = args.ckpt_timelabel
else:
    localtime = time.localtime(time.time())
    time_label = str(localtime.tm_year) + '_' + str(localtime.tm_mon) + '_' + str(localtime.tm_mday) + '_' + str(localtime.tm_hour) + '_' + str(localtime.tm_min)
args.ckpt_path = os.path.join(args.ckpt_path, args.dataset_name, args.model_name, time_label)
if not os.path.exists(args.ckpt_path):     # test, not exists
    os.makedirs(args.ckpt_path)

logging.config.fileConfig("./logging.conf")

# create logger
log = logging.getLogger()

# no need to append an extra feature axis if data is multichannel
add_feat_axis = not args.multichannel

train_files = read_mat_list(args.data_path, train=True)
val_files = read_mat_list(args.data_path, train=False)

train_generator = generators.slice_to_slice(train_files, batch_size=args.batch_size, bidir=args.bidir, add_feat_axis=add_feat_axis)
val_generator = generators.slice_to_slice(val_files, batch_size=args.batch_size, bidir=args.bidir, add_feat_axis=add_feat_axis)

inshape = next(train_generator)[0][0].shape[2:]
args.input_height = inshape[0]
args.input_width = inshape[1]


def calc_confidence_interval(samples, confidence_value=0.95):
    # samples should be a numpy array
    if type(samples) is list:
        samples = np.asarray(samples)
    assert isinstance(samples, np.ndarray), 'args: samples {} should be np.array'.format(samples)
    # print('Results List:', samples)
    stat_accu = st.t.interval(confidence_value, len(samples) - 1, loc=np.mean(samples), scale=st.sem(samples))
    center = (stat_accu[0] + stat_accu[1]) / 2
    deviation = (stat_accu[1] - stat_accu[0]) / 2
    return center, deviation


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, num_top=10):
        self.reset()
        _array = np.zeros(shape=(num_top)) + 0.01
        self.top_list = _array.tolist()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def top_update_calc(self, val):
        # update the lowest or NOT
        if val > self.top_list[0]:
            self.top_list[0] = val
            # [lowest, ..., highest]
            self.top_list.sort()
        # update mean, deviation
        mean, deviation = calc_confidence_interval(self.top_list)
        best = self.top_list[-1]
        return mean, deviation, best


# For updating learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu
# device handling
gpus = args.gpu.split(',')
nb_gpus = len(gpus)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

debug = True


# if nb_gpus > 1:
#     # use multiple GPUs via DataParallel
#     model = torch.nn.DataParallel(model)
#     model.save = model.module.save
def train():

    model = DisQ(beta_dim = 3,
                 theta_dim = 2,
                 train_sample = 'st_gumbel_softmax',
                 valid_sample = 'argmax',
                 pretrained_model = None,
                 initial_temp = args.initial_temp,
                 anneal_rate = args.anneal_rate,
                 device = device,
                 fine_tune = args.fine_tune)

    model.initialize_training(out_dir=args.ckpt_path, lr=args.lr)

    optim_beta_encoder = torch.optim.Adam(model.beta_encoder.parameters(), lr=args.lr)
    optim_theta_encoder = torch.optim.Adam(model.theta_encoder.parameters(), lr=args.lr)
    optim_decoder = torch.optim.Adam(model.decoder.parameters(), lr=args.lr)
    optim_decoder_affine = torch.optim.Adam(model.decoder_affine.parameters(), lr=args.lr)
    optim_da_beta = torch.optim.Adam(model.da_beta.parameters(), lr=args.lr)
    optim_da_theta = torch.optim.Adam(model.da_theta.parameters(), lr=args.lr)
    optim_af_theta = torch.optim.Adam(model.af_theta.parameters(), lr=args.lr)
    temp_sched = TemperatureAnneal(initial_temp=model.initial_temp, anneal_rate=model.anneal_rate, min_temp=0.5, device=model.device)

    losses = AverageMeter()
    losses_recon = AverageMeter()
    losses_beta = AverageMeter()
    losses_kld = AverageMeter()
    losses_per = AverageMeter()
    epoch_time = AverageMeter()

    if not debug:
        vis = Visualizer(env=args.vis_env, port=8097)

    cudnn.benchmark = True

    monitor_metric_best = 1e8

    end = time.time()

    # training loops
    # CcM = torch.FloatTensor([3]).to(device)
    # CcB = torch.FloatTensor([238]).to(device)
    CcA = torch.FloatTensor([1000.0]).to(device)
    CcB = torch.FloatTensor([0.002]).to(device)
    for epoch in range(args.initial_epoch, args.epochs):
        # model.train()

        Cc = CcA * torch.exp(CcB * epoch)

        model.beta_encoder.train()
        model.theta_encoder.train()
        model.decoder.train()
        model.decoder_affine.train()
        model.da_beta.train()
        model.da_theta.train()
        model.af_theta.train()

        for step in range(args.steps_per_epoch):

            # generate inputs (and true outputs) and convert them to tensors
            inputs, y_true = next(train_generator)

            inputs = [torch.from_numpy(d).to(device).float() for d in inputs]
            y_true = [torch.from_numpy(d).to(device).float() for d in y_true]

            betas, logits = model.cal_beta(inputs, model.train_sample, temp_sched)
            thetas, mus, logvars = model.cal_theta(inputs)

            amus, alogvars = model.affine_theta(thetas)

            # affine_zs = model.anatomy_transformation(amus, alogvars, betas)
            #
            # cat_betas = model.cat_betaz(betas, affine_zs)

            cat_betas = model.cat_betamu(betas, amus)

            rec_imgs, img_ids = model.decode_aff(cat_betas, mus, swap_beta=False, self_rec=True)

            rec_loss = 0
            per_loss = 0
            for rec_img, img_id in zip(rec_imgs, img_ids):
                compare_img = inputs[img_id]
                # l1 loss
                rec_loss += model.l1_loss(rec_img, compare_img).mean()
                # perceptual loss
                rec_feature = model.vgg(rec_img.repeat(1,3,1,1)).relu2_2
                tar_feature = model.vgg(compare_img.repeat(1,3,1,1)).relu2_2
                per_loss += model.l1_loss(rec_feature, tar_feature).mean()
            per_loss = per_loss / len(rec_imgs)
            rec_loss = rec_loss / len(rec_imgs)

            kld_loss = 0.0
            # for mu, logvar in zip(mus, logvars):
            #     kld_loss += model.kld_loss(mu, logvar).mean()
            # kld_loss = kld_loss / len(mus)
            mus_2 = (mus[0] * mus[0]) + (mus[1] * mus[1])

            kld_loss = model.l1_loss(mus_2, Cc.repeat(mus_2.shape[0],mus_2.shape[1],mus_2.shape[2],mus_2.shape[3])).mean()

            # beta_loss = model.l1_loss(betas[0], betas[1]).mean()
            beta_loss = model.shape_loss(betas[0], betas[1]).mean()

            loss = 2*rec_loss + 2e-2*beta_loss + 1e-8*kld_loss + 3e-2*per_loss


            losses.update(loss.item(), args.batch_size)
            losses_recon.update(rec_loss.item(), args.batch_size)
            losses_beta.update(beta_loss.item(), args.batch_size)
            losses_kld.update(kld_loss.item(), args.batch_size)
            losses_per.update(per_loss.item(), args.batch_size)

            if torch.isnan(loss):
                pdb.set_trace()

            # backpropagate
            optim_beta_encoder.zero_grad()
            optim_theta_encoder.zero_grad()
            optim_decoder.zero_grad()
            optim_decoder_affine.zero_grad()
            optim_da_beta.zero_grad()
            optim_da_theta.zero_grad()
            optim_af_theta.zero_grad()
            loss.backward()
            optim_beta_encoder.step()
            optim_theta_encoder.step()
            optim_decoder.step()
            optim_decoder_affine.step()
            optim_da_beta.step()
            optim_da_theta.step()
            optim_af_theta.step()
            temp_sched.step()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            for name, param in model.named_parameters():
                try:
                    if not torch.isfinite(param.grad).all():
                        pdb.set_trace()
                except:
                    continue

            # print(name, torch.isfinite(param.grad).all())


        # visualization and evaluation
        if not debug and (epoch + 1) % 1 == 0:
            x_0   = inputs[0].detach()
            x_1   = inputs[1].detach()
            s_0   = betas[0].detach()
            s_1   = betas[1].detach()
            x_00  = rec_imgs[0].detach()
            x_01  = rec_imgs[1].detach()  # 0: contrast 1: shape
            x_10  = rec_imgs[2].detach()
            x_11  = rec_imgs[3].detach()

            # for visdom vis
            x_0   =  1 * (x_0 + 0)
            x_1   =  1 * (x_1 + 0)
            s_0   =  1 * (s_0 + 0)
            s_1   =  1 * (s_1 + 0)
            x_00  =  1 * (x_00 + 0)
            x_01  =  1 * (x_01 + 0)
            x_10  =  1 * (x_10 + 0)
            x_11  =  1 * (x_11 + 0)

            vis.img(name='x_0', img_=x_0[0, 0])
            vis.img(name='x_1', img_=x_1[0, 0])
            vis.img(name='s_00', img_=s_0[0, 0])
            vis.img(name='s_10', img_=s_1[0, 0])
            vis.img(name='s_01', img_=s_0[0, 1])
            vis.img(name='s_11', img_=s_1[0, 1])
            vis.img(name='s_02', img_=s_0[0, 2])
            vis.img(name='s_12', img_=s_1[0, 2])
            # vis.img(name='s_02', img_=s_0[0, 2])
            # vis.img(name='s_12', img_=s_1[0, 2])
            vis.img(name='x_00', img_=x_00[0, 0])
            vis.img(name='x_01', img_=x_01[0, 0])
            vis.img(name='x_10', img_=x_10[0, 0])
            vis.img(name='x_11', img_=x_11[0, 0])

        # get compute time
        epoch_time.update(time.time() - end)
        end = time.time()
        # print epoch info
        log.info(
            'Epoch [{}], Start [{}], Step [{}/{}], Loss: {:.4f}, Time [{batch_time.val:.3f}({batch_time.avg:.3f})]'
                .format(epoch + 1, args.initial_epoch, epoch + 1, args.epochs, loss.item(),
                        batch_time=epoch_time))

        if not debug:
            vis.plot_multi_win(
                dict(
                    losses_total=losses.avg,
                    losses_recon=losses_recon.avg,
                    losses_kld=losses_kld.avg,
                    losses_beta=losses_beta.avg,
                    losses_per=losses_per.avg,
                ))

        # Validata
        if (epoch + 1) % 5 == 0:
            # model.eval()

            model.beta_encoder.eval()
            model.theta_encoder.eval()
            model.decoder.eval()
            model.decoder_affine.eval()
            model.da_beta.eval()
            model.da_theta.eval()
            model.af_theta.eval()
            loss_all_dict = {'recon': 0., 'kld': 0.,
                             'beta': 0., 'per': 0., 'sim_z': 0., 'all': 0.}

            with torch.no_grad():
                val_iter = 50
                for ite in range(val_iter):       ## any number

                    inputs, y_true = next(val_generator)

                    inputs = [torch.from_numpy(d).to(device).float() for d in inputs]
                    y_true = [torch.from_numpy(d).to(device).float() for d in y_true]

                    betas, logits = model.cal_beta(inputs, model.valid_sample, temp_sched)
                    thetas, mus, logvars = model.cal_theta(inputs)

                    amus, alogvars = model.affine_theta(thetas)

                    # affine_zs = model.anatomy_transformation(amus, alogvars, betas)
                    #
                    # cat_betas = model.cat_betaz(betas, affine_zs)

                    cat_betas = model.cat_betamu(betas, amus)

                    rec_imgs, img_ids = model.decode_aff(cat_betas, mus, swap_beta=False, self_rec=True)

                    rec_loss = 0
                    per_loss = 0
                    for rec_img, img_id in zip(rec_imgs, img_ids):
                        compare_img = inputs[img_id]
                        # l1 loss
                        rec_loss += model.l1_loss(rec_img, compare_img).mean()
                        # perceptual loss
                        rec_feature = model.vgg(rec_img.repeat(1,3,1,1)).relu2_2
                        tar_feature = model.vgg(compare_img.repeat(1,3,1,1)).relu2_2
                        per_loss += model.l1_loss(rec_feature, tar_feature).mean()
                    per_loss = per_loss / len(rec_imgs)
                    rec_loss = rec_loss / len(rec_imgs)

                    kld_loss = 0.0
                    # for mu, logvar in zip(mus, logvars):
                    #     kld_loss += model.kld_loss(mu, logvar).mean()
                    # kld_loss = kld_loss / len(mus)
                    # kld_loss = (thetas[0] * thetas[0]).mean() + (thetas[1] * thetas[1]).mean()
                    mus_2 = (mus[0] * mus[0]) + (mus[1] * mus[1])

                    kld_loss = model.l1_loss(mus_2, Cc.repeat(mus_2.shape[0],mus_2.shape[1],mus_2.shape[2],mus_2.shape[3])).mean()

                    # beta_loss = model.l1_loss(betas[0], betas[1]).mean()

                    beta_loss = model.shape_loss(betas[0], betas[1]).mean()

                    loss = 2*rec_loss + 2e-2*beta_loss + 1e-8*kld_loss + 3e-2*per_loss
                    #loss = 2*rec_loss + 2e-2*beta_loss + 3e-2*per_loss

                    loss_all_dict['recon'] += rec_loss.item()
                    loss_all_dict['per'] += per_loss.item()
                    loss_all_dict['kld'] += kld_loss.item()
                    loss_all_dict['beta'] += beta_loss.item()
                    loss_all_dict['all'] += loss.item()

                monitor_metric = loss_all_dict['recon'] / val_iter
                print(monitor_metric)

                # save ckp
                is_best = False
                if monitor_metric <= monitor_metric_best:
                    is_best = True
                    monitor_metric_best = monitor_metric if is_best is True else monitor_metric_best
                state = {'epoch': epoch, 'monitor_metric': monitor_metric, 'optim_beta_encoder': optim_beta_encoder.state_dict(),
                        'optim_theta_encoder': optim_theta_encoder.state_dict(), 'optim_decoder': optim_decoder.state_dict(),
                        'optim_da_beta': optim_da_beta.state_dict(), 'temp_sched': temp_sched.state_dict(),
                         'model': model.state_dict()}

                save_checkpoint(state, is_best, args.ckpt_path)



if __name__ == '__main__':
    train()  ## train and eval

