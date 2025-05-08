from __future__ import print_function, division
import os
import argparse
import time

import torch.nn as nn
from skimage import io
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from datasets_mask import __datasets__
from models.model.MambaStereo import MambaStereo
from utils import *
from utils.KittiColormap import *

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='MambaStereo')
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--dataset', default='kitti', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', default='../../datasets/kitti2015/', help='data path')
parser.add_argument('--testlist', default='./filenames/kitti15_submit.txt', help='testing list')
parser.add_argument('--loadckpt', default='', help='load the weights from a specific checkpoint')
parser.add_argument('--colored', default=0, help='save colored or save for benchmark submission')

# parse arguments
args = parser.parse_args()

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
test_dataset = StereoDataset(args.datapath, args.testlist, False)
TestImgLoader = DataLoader(test_dataset, 1, shuffle=False, num_workers=4, drop_last=False)

# model, optimizer
model = MambaStereo(args.maxdisp)
# model = __models__['acvnet'](192, False, False)
model.cuda()

# load parameters
print("Loading model {}".format(args.loadckpt))
state_dict = torch.load(args.loadckpt)
model.load_state_dict(state_dict['model'], strict=True)


def test(args):
    print("Generating the disparity maps...")

    os.makedirs('predictions/mask15/', exist_ok=True)

    for batch_idx, sample in enumerate(TestImgLoader):

        disp_est_tn = test_sample(sample)
        disp_est_np = tensor2numpy(disp_est_tn)
        top_pad_np = tensor2numpy(sample["top_pad"])
        right_pad_np = tensor2numpy(sample["right_pad"])
        left_filenames = sample["left_filename"]

        for disp_est, top_pad, right_pad, fn in zip(disp_est_np, top_pad_np, right_pad_np, left_filenames):

            assert len(disp_est.shape) == 2

            disp_est = np.array(disp_est[top_pad:, :-right_pad], dtype=np.float32)
            name = fn.split('/')
            fn = os.path.join("predictions/mask15", '_'.join(name[2:]))

            if float(args.colored) == 1:
                disp_est = kitti_colormap(disp_est)
                cv2.imwrite(fn, disp_est)
            else:
                disp_est = np.round(disp_est * 256).astype(np.uint16)
                io.imsave(fn, disp_est)

    print("Done!")


@make_nograd_func
def test_sample(sample):
    model.eval()
    # start_time = time.time()
    disp_ests = model(sample['left'].cuda(), sample['right'].cuda())
    # print('time = {:3f}'.format(time.time()-start_time))
    return disp_ests[-1]


if __name__ == '__main__':
    test(args)
