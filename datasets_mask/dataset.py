import os
import random
import numpy as np
import torch
from PIL import Image
import matplotlib
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from datasets_mask.data_io import get_transform, get_transform_aug, read_all_lines, pfm_imread


class SceneFlowDataset(Dataset):
    def __init__(self, datapath, list_filename, training):
        self.datapath = datapath
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(list_filename)
        self.training = training

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        disp_images = [x[2] for x in splits]
        return left_images, right_images, disp_images

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        data, scale = pfm_imread(filename)
        data = np.ascontiguousarray(data, dtype=np.float32)
        return data

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))
        disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]))

        if self.training:
            w, h = left_img.size
            crop_w, crop_h = 512, 256

            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)
            # d = random.randint(0, 20)

            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]

            if random.random() > 0.5:
                saturation = np.random.uniform(0.8, 1.2)
                # hue = np.random.uniform(-0.1, 0.1)
                brightness = np.random.uniform(0.5, 1.5)
                gamma = np.random.uniform(0.8, 1.2)
                contrast_factor = np.random.uniform(0.8, 1.2)
                processed = get_transform_aug(contrast_factor, gamma, brightness, saturation)
            else:
                processed = get_transform_aug(1, 1, 1, 1)
            left_img = processed(left_img)
            right_img = processed(right_img)
            # print(left_img.size)
            left_img = np.asarray(left_img)
            right_img = np.asarray(right_img).copy()
            right_img.flags.writeable = True
            left_img = get_transform()(left_img)
            right_img = get_transform()(right_img)
            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity,
                    }

        else:
            w, h = left_img.size
            crop_w, crop_h = 960, 512

            left_img = left_img.crop((w - crop_w, h - crop_h, w, h))
            right_img = right_img.crop((w - crop_w, h - crop_h, w, h))
            disparity = disparity[h - crop_h:h, w - crop_w: w]

            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity,
                    "top_pad": 0,
                    "right_pad": 0,
                    "left_filename": self.left_filenames[index]}


class KITTIDataset(Dataset):
    def __init__(self, datapath, list_filename, training):
        self.datapath = datapath
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(list_filename)
        self.training = training
        if self.training:
            assert self.disp_filenames is not None

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        if len(splits[0]) == 2:  # ground truth not available
            return left_images, right_images, None
        else:
            disp_images = [x[2] for x in splits]
            return left_images, right_images, disp_images

    def load_image(self, filename):
        image = Image.open(filename).convert('RGB')
        return image

    def load_disp(self, filename):
        data = Image.open(filename)
        data = np.array(data, dtype=np.float32) / 256.
        return data

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))

        if self.disp_filenames:  # has disparity ground truth
            disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]))
        else:
            disparity = None

        if self.training:
            w, h = left_img.size
            crop_w, crop_h = 512, 256
            # crop_w, crop_h = 256, 128
            # crop_w, crop_h = 1024, 256
            max_cropw = w - crop_w
            max_croph = h - crop_h
            x1 = random.randint(0, max_cropw)
            y1 = random.randint(0, max_croph)

            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]
            if random.random() > 0.6:
                brightness = np.random.uniform(0.8, 1.2)
                gamma = np.random.uniform(0.8, 1.2)
                contrast_factor = np.random.uniform(0.8, 1.2)
                saturation = np.random.uniform(0.8, 1.2)
                # hue = np.random.uniform(-0.1, 0.1)
                # brightness = np.random.uniform(0.7, 1.3)
                # gamma = np.random.uniform(0.7, 1.3)
                # contrast_factor = np.random.uniform(0.7, 1.3)
                processed = get_transform_aug(contrast_factor, gamma, brightness, saturation)
            else:
                processed = get_transform_aug(1, 1, 1, 1)
            left_img = processed(left_img)
            right_img = processed(right_img)

            left_img = np.asarray(left_img)
            right_img = np.asarray(right_img).copy()
            right_img.flags.writeable = True
            # print("mask")
            # if random.random() > 0.2:
            sx = int(np.random.uniform(35, 100))
            sy = int(np.random.uniform(25, 75))
            cx = int(np.random.uniform(sx, right_img.shape[0] - sx))
            cy = int(np.random.uniform(sy, right_img.shape[1] - sy))
            right_img[cx - sx:cx + sx, cy - sy:cy + sy] = np.mean(np.mean(right_img, 0), 0)[np.newaxis, np.newaxis]

            left_img = get_transform()(left_img)
            right_img = get_transform()(right_img)
            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity}
        else:
            w, h = left_img.size

            left_img = (get_transform()(left_img)).numpy()
            right_img = (get_transform()(right_img)).numpy()

            # pad to size 1248x384
            # top_pad = 384 - h
            top_pad = 528 - h
            right_pad = 1248 - w
            assert top_pad > 0 and right_pad > 0
            # pad images
            left_img = np.lib.pad(left_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            right_img = np.lib.pad(right_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant',
                                   constant_values=0)
            # pad disparity gt
            if disparity is not None:
                assert len(disparity.shape) == 2
                disparity = np.lib.pad(disparity, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)

            if disparity is not None:
                return {"left": left_img,
                        "right": right_img,
                        "disparity": disparity,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "left_filename": self.left_filenames[index]}
            else:
                return {"left": left_img,
                        "right": right_img,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "left_filename": self.left_filenames[index],
                        "right_filename": self.right_filenames[index]}


class DrivingStereoDataset(Dataset):
    def __init__(self, datapath, list_filename, training):
        self.datapath = datapath
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(list_filename)
        self.training = training

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        disp_images = [x[2] for x in splits]
        return left_images, right_images, disp_images

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        data = Image.open(filename)
        data = np.array(data, dtype=np.float32) / 256.
        return data

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))
        disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]))

        if self.training:
            w, h = left_img.size  # (881, 400)
            crop_w, crop_h = 512, 256

            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)

            # random crop
            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]

            # to tensor, normalize
            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity}

        else:
            w, h = left_img.size
            crop_w, crop_h = 880, 400

            left_img = left_img.crop((w - crop_w, h - crop_h, w, h))
            right_img = right_img.crop((w - crop_w, h - crop_h, w, h))
            disparity = disparity[h - crop_h:h, w - crop_w: w]

            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity,
                    "top_pad": 0,
                    "right_pad": 0,
                    "left_filename": self.left_filenames[index]}
