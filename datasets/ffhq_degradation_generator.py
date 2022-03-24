import cv2
import math
import numpy as np
import torch
import torch.utils.data as data
from basicsr.data import degradations as degradations
from basicsr.data.data_util import paths_from_folder
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from torchvision.transforms.functional import (adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation,
                                               normalize)



class FFHQDegradationDataset(data.Dataset):
    """FFHQ dataset for GFPGAN.

    It reads high resolution images, and then generate low-quality (LQ) images on-the-fly.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            io_backend (dict): IO backend type and other kwarg.
            mean (list | tuple): Image mean.
            std (list | tuple): Image std.
            use_hflip (bool): Whether to horizontally flip.
            Please see more options in the codes.
    """

    def __init__(self, opt, degradation):
        super(FFHQDegradationDataset, self).__init__()
        self.opt = opt
        self.degradation = degradation

        self.base_size = opt['base_size']
        self.gt_folder = opt['dataroot_gt']
        self.mean = opt['mean']
        self.std = opt['std']
        self.out_size = opt['out_size']

        # disk backend: scan file list from a folder
        self.paths = paths_from_folder(self.gt_folder)

        # degradation configurations
        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']
        self.blur_sigma = opt['blur_sigma']
        self.downsample_range = opt['downsample_range']
        self.noise_range = opt['noise_range']
        self.jpeg_range = opt['jpeg_range']

        # color jitter
        self.color_jitter_prob = opt.get('color_jitter_prob')
        self.color_jitter_pt_prob = opt.get('color_jitter_pt_prob')
        self.color_jitter_shift = opt.get('color_jitter_shift', 20)
        # to gray
        self.gray_prob = opt.get('gray_prob')
        self.color_jitter_shift /= 255.

    @staticmethod
    def color_jitter(img, shift):
        """jitter color: randomly jitter the RGB values, in numpy formats"""
        jitter_val = np.random.uniform(-shift, shift, 3).astype(np.float32)
        img = img + jitter_val
        img = np.clip(img, 0, 1)
        return img

    @staticmethod
    def color_jitter_pt(img, brightness, contrast, saturation, hue):
        """jitter color: randomly jitter the brightness, contrast, saturation, and hue, in torch Tensor formats"""
        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and brightness is not None:
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = adjust_brightness(img, brightness_factor)

            if fn_id == 1 and contrast is not None:
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = adjust_contrast(img, contrast_factor)

            if fn_id == 2 and saturation is not None:
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = adjust_saturation(img, saturation_factor)

            if fn_id == 3 and hue is not None:
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = adjust_hue(img, hue_factor)
        return img


    def __getitem__(self, index):
        # load gt image
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = self.paths[index]
        img_gt = cv2.imread(gt_path)

        img_gt = cv2.resize(img_gt, (self.base_size, self.base_size))
        img_gt = img_gt.astype(np.float32)
        img_gt = img_gt / 255.

        # random horizontal flip
        img_gt, status = augment(img_gt, hflip=self.opt['use_hflip'], rotation=False, return_status=True)
        h, w, _ = img_gt.shape

        # ------------------------ generate lq image ------------------------ #
        # blur
        kernel = degradations.random_mixed_kernels(
            self.kernel_list,
            self.kernel_prob,
            self.blur_kernel_size,
            self.blur_sigma,
            self.blur_sigma, [-math.pi, math.pi],
            noise_range=None)
        img_lq = cv2.filter2D(img_gt, -1, kernel)

        # downsample
        scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
        print(img_lq.shape, img_lq.min(), img_lq.max())
        # noise
        if self.noise_range is not None:
            img_lq = degradations.random_add_gaussian_noise(img_lq, self.noise_range)
        print(img_lq.shape, img_lq.min(), img_lq.max())
        # return img_gt, img_lq
        # jpeg compression
        if self.jpeg_range is not None and False:
            img_lq = degradations.random_add_jpg_compression(img_lq, self.jpeg_range)

        # resize to original size
        img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)

        # random color jitter (only for lq)
        if self.color_jitter_prob is not None and (np.random.uniform() < self.color_jitter_prob):
            img_lq = self.color_jitter(img_lq, self.color_jitter_shift)
        # random to gray (only for lq)
        if self.gray_prob and np.random.uniform() < self.gray_prob:
            img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGR2GRAY)
            img_lq = np.tile(img_lq[:, :, None], [1, 1, 3])
            if self.opt.get('gt_gray'):  # whether convert GT to gray images
                img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
                img_gt = np.tile(img_gt[:, :, None], [1, 1, 3])  # repeat the color channels

        print(img_lq.shape, img_lq.min(), img_lq.max())
        # img_lq = (img_lq * 255.).astype(np.int8)
        # img_lq[img_lq] >

        img_gt = (img_gt * 255.0) # .astype(np.int8)
        img_lq = (img_lq * 255.0) # .astype(np.int8)
        print(img_lq.shape, img_lq.min(), img_lq.max())

        return img_gt, img_lq
        # return {'lq': img_lq, 'gt': img_gt, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)
