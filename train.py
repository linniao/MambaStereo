from __future__ import print_function, division
import argparse
import gc
import os
import time
import datetime
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
# from datasets_mask import __datasets__
from datasets import __datasets__
from loss import model_loss
from models.model.MambaStereo import MambaStereo

from utils import *

# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='MambaStereo')
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--lr', type=float, default=0.0005, help='base learning rate')
parser.add_argument('--batch_size', type=int, default=2, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=2, help='testing batch size')
parser.add_argument('--resume', default='', action='store_true', help='continue training the model')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--dataset', default='kitti', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', default='../../datasets/kitti2015', help='data path')
parser.add_argument('--trainlist', default='filenames/kitti15_train_all.txt', help='training list')
parser.add_argument('--testlist', default='filenames/kitti15_test.txt', help='testing list')
parser.add_argument('--ckptdir', default='ckpts/',
                    help='the directory to save checkpoints')
parser.add_argument('--loadckpt', default='',
                    help='load the weights from a specific checkpoint')
parser.add_argument('--epochs', type=int, default=600, help='number of epochs to train')
parser.add_argument('--summary_freq', type=int, default=200, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=100, help='the frequency of saving checkpoint')

# parser.add_argument('--dataset', default='sceneflow', help='dataset name', choices=__datasets__.keys())
# parser.add_argument('--datapath', default='../../datasets/SceneFlow/', help='data path')
# parser.add_argument('--trainlist', default='./filenames/sceneflow_cleanpass_train.txt', help='training list')
# parser.add_argument('--testlist', default='./filenames/sceneflow_cleanpass_test.txt', help='testing list')
# parser.add_argument('--ckptdir', default='ckpts/', help='the directory to save checkpoints')
# parser.add_argument('--loadckpt', default='', help='load the weights from the checkpoint')
# parser.add_argument('--epochs', type=int, default=18, help='number of epochs to train')
# parser.add_argument('--summary_freq', type=int, default=500, help='the frequency of saving summary')
# parser.add_argument('--save_freq', type=int, default=2, help='the frequency of saving checkpoint')


# parse arguments, set seeds
args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
os.makedirs(args.ckptdir, exist_ok=True)

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
train_dataset = StereoDataset(args.datapath, args.trainlist, True)
test_dataset = StereoDataset(args.datapath, args.testlist, False)
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=8, pin_memory=True,
                            drop_last=True)
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=8, pin_memory=True,
                           drop_last=False)

# model, optimizer
model = MambaStereo(args.maxdisp)
model = model.cuda()
# print(model)
# optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
optimizer = optim.AdamW(params=model.parameters(), weight_decay=0.01, lr=args.lr)

# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 16], 0.5)

scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                args.lr,
                                                epochs=args.epochs,
                                                steps_per_epoch=1,
                                                cycle_momentum=True,
                                                base_momentum=0.85,
                                                max_momentum=0.95,
                                                last_epoch=-1,
                                                div_factor=10,
                                                final_div_factor=20)

start_epoch = 0
if args.resume:
    print("resume")
    # find all checkpoints file and sort according to epoch id
    all_saved_ckpts = [fn for fn in os.listdir(args.ckptdir)[1:] if fn.endswith(".ckpt")]
    all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the latest checkpoint file
    loadckpt = os.path.join(args.ckptdir, all_saved_ckpts[-1])
    print("Loading the latest model in logdir: {}".format(loadckpt))
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    # scheduler.load_state_dict(state_dict['scheduler'])
    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    # load the checkpoint file specified by args.loadckpt
    print("Loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'], strict=True)
    '''state_dict = torch.load(args.loadckpt)
    model_dict = model.state_dict()
    pre_dict = {}
    for k, v in state_dict['model'].items():
        if 'module.' in k:
            k = k.replace('module.', '')
            pre_dict[k] = v
    model_dict.update(pre_dict)
    model.load_state_dict(model_dict, strict=True)'''
    '''model_dict = model.state_dict()
    pre_dict = {}
    for k, v in state_dict['model'].items():
        if 'gwc' in k:
            print(k)
            pre_dict[k] = v
    model_dict.update(pre_dict)
    model.load_state_dict(model_dict)'''
print("Start at epoch {}".format(start_epoch))

# create summary logger
prefix = '15+12_15_100%mask'
logger = SummaryWriter(comment='{}-lr{}-e{}-bs{}'.format(prefix, args.lr, args.epochs, args.batch_size), flush_secs=60)


def train():
    best_checkpoint_loss = 1.0
    for epoch_idx in range(start_epoch, args.epochs):
        losses = AverageMeter()
        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = train_sample(sample, epoch_idx, compute_metrics=do_summary)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            loss = tensor2float(loss)
            losses.update(loss)
            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)
                save_images(logger, 'train', image_outputs, global_step)
            del scalar_outputs, image_outputs
            print('Epoch {}/{}, Iter {}/{}, train loss = {:.3f} ({:.3f}), time = {:.3f} (ETA:{})'.format
                  (epoch_idx,
                   args.epochs,
                   batch_idx,
                   len(TrainImgLoader),
                   loss,
                   losses.mean(),
                   time.time() - start_time,
                   str(datetime.timedelta(seconds=int((time.time() - start_time) * (len(TrainImgLoader) - batch_idx))))
                   )
                  )
        # saving checkpoints
        if (epoch_idx + 1) % args.save_freq == 0:
            checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                               'scheduler': scheduler.state_dict()}
            torch.save(checkpoint_data, "{}/checkpoint_{:0>6}.ckpt".format(args.ckptdir, epoch_idx))
        scheduler.step()
        gc.collect()

        # if True:
        # testing
        if epoch_idx >= 0:
            avg_test_scalars = AverageMeterDict()
            for batch_idx, sample in enumerate(TestImgLoader):
                global_step = len(TestImgLoader) * epoch_idx + batch_idx
                start_time = time.time()
                do_summary = global_step % args.summary_freq == 0
                loss, scalar_outputs, image_outputs = test_sample(sample, epoch_idx, compute_metrics=do_summary)
                if do_summary:
                    save_scalars(logger, 'test', scalar_outputs, global_step)
                    save_images(logger, 'test', image_outputs, global_step)
                avg_test_scalars.update(scalar_outputs)
                del scalar_outputs, image_outputs
                print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch_idx, args.epochs,
                                                                                         batch_idx,
                                                                                         len(TestImgLoader), loss,
                                                                                         time.time() - start_time))
            avg_test_scalars = avg_test_scalars.mean()

            save_scalars(logger, 'fulltest', avg_test_scalars, len(TrainImgLoader) * (epoch_idx + 1))
            print("avg_test_scalars", avg_test_scalars)

            # saving new best checkpoint
            if avg_test_scalars['loss'] <= best_checkpoint_loss:
                best_checkpoint_loss = avg_test_scalars['loss']
                print("Overwriting best checkpoint")
                checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
                torch.save(checkpoint_data, "{}/best.ckpt".format(args.ckptdir))

            gc.collect()


# train one sample
def train_sample(sample, epoch_idx, compute_metrics=False):
    model.train()

    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()
    # print(disp_gt.max())

    disp_ests = model(imgL, imgR)

    # print(f'disp_ests:{len(disp_ests)}')

    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    loss = model_loss(disp_ests, disp_gt, mask, epoch_idx, mode='train')

    scalar_outputs = {"loss": loss, "LR": optimizer.param_groups[0]['lr']}
    image_outputs = {"disp_est": disp_ests[0:4], "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}
    if compute_metrics:
        with torch.no_grad():
            image_outputs["errormap"] = [disp_error_image_func(disp_est, disp_gt) for disp_est in disp_ests[0:4]]
            scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests[0:4]]
            scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests[0:4]]
            scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests[0:4]]
            scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests[0:4]]
            scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests[0:4]]

    return loss, tensor2float(scalar_outputs), image_outputs


# test one sample
@make_nograd_func
def test_sample(sample, epoch_idx, compute_metrics=True):
    model.eval()

    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()
    # print("img", imgR.shape)
    disp_ests = model(imgL, imgR)
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    loss = model_loss(disp_ests, disp_gt, mask, epoch_idx, mode='test')

    scalar_outputs = {"loss": loss}
    image_outputs = {"disp_est": disp_ests[0:4], "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}

    scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests[0:4]]
    scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests[0:4]]
    scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests[0:4]]
    scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests[0:4]]
    scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests[0:4]]

    if compute_metrics:
        image_outputs["errormap"] = [disp_error_image_func(disp_est, disp_gt) for disp_est in disp_ests[0:4]]

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


if __name__ == '__main__':
    train()
