import argparse
import cv2
import glob
import numpy as np
import os
import torch
from basicsr.utils import imwrite
from datasets.ffhq_degradation_generator import FFHQDegradationDataset
from easydict import EasyDict
import timm

def get_dataset_opt():
    opt = EasyDict()

    opt.dataroot_gt = 'FFHQ'

    opt.base_size = 512
    opt.use_hflip = True
    opt.mean = (0.5, 0.5, 0.5)
    opt.std = (0.5, 0.5, 0.5)
    opt.out_size = 512

    opt.blur_kernel_size = 5
    opt.kernel_list = ('iso', 'aniso')
    opt.kernel_prob = (0.5, 0.5)
    opt.blur_sigma = (0.1, 10)
    opt.downsample_range = (0.8, 4)
    opt.noise_range = (0, 5)
    opt.jpeg_range = (60, 100)

    # color jitter and gray
    opt.color_jitter_prob = 0.3
    opt.color_jitter_shift = 20
    opt.color_jitter_pt_prob = 0.3
    opt.gray_prob = 0.01
    return opt


def main():
    dataset_opt = get_dataset_opt()
    dataset = FFHQDegradationDataset(dataset_opt, False)

    os.makedirs('degradation', exist_ok=True)

    for i in range(20):
        img_hq, img_lq = dataset[i]
        cv2.imwrite(f'degradation/{i}_hq.png', img_hq)
        cv2.imwrite(f'degradation/{i}_lq.png', img_lq)

if __name__ == '__main__':
    main()