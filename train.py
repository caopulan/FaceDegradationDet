import argparse
import cv2
import glob
import numpy as np
import os
import torch
from basicsr.utils import imwrite
from datasets.ffhq_degradation_dataset import FFHQDegradationDataset
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
    epochs = 100
    batch_size = 16

    dataset_opt = get_dataset_opt()
    hq_dataset = FFHQDegradationDataset(dataset_opt, False)
    lq_dataset = FFHQDegradationDataset(dataset_opt, True)
    hq_data_loader = torch.utils.data.DataLoader(hq_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    lq_data_loader = torch.utils.data.DataLoader(lq_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    print('images:', len(hq_dataset))

    model = timm.models.convnext.convnext_tiny(True, num_classes=1).cuda()
    model.train()

    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    for epoch in range(epochs):
        for iter, (hq_data, lq_data) in enumerate(zip(hq_data_loader, lq_data_loader)):
            targets = torch.cat([torch.ones(hq_data.shape[0]), torch.zeros(lq_data.shape[0])]).cuda()
            input_images = torch.cat([hq_data, lq_data], dim=0).cuda()

            optim.zero_grad()
            output = model(input_images)
            output = output[:, 0]
            loss = criterion(output, targets)
            loss.backward()
            optim.step()

            acc = (targets == (output > 0.5)).sum().item() / (hq_data.shape[0] + lq_data.shape[0])
            print(epoch, iter, acc, loss.item())
            torch.save(model.state_dict(), 'model.pth')

if __name__ == '__main__':
    main()