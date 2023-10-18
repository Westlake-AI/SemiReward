# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import torchvision
import numpy as np
import math
import pandas as pd
from PIL import Image

from torchvision import transforms
from .datasetbase import BasicDataset
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation
from semilearn.datasets.utils import split_ssl_data


class AgeDBIDataset(BasicDataset):

    def __init__(self,
                 alg,
                 data,
                 targets=None,
                 num_classes=None,
                 transform=None,
                 is_ulb=False,
                 strong_transform=None,
                 onehot=False,
                 *args, 
                 **kwargs):
        super(AgeDBIDataset, self).__init__(alg=alg, data=data, targets=targets, num_classes=num_classes,
                transform=transform, is_ulb=is_ulb, strong_transform=strong_transform, onehot=onehot, *args, **kwargs)
        data_dir = kwargs.get('data_dir', '')
        self.data = [os.path.join(data_dir, data) for data in self.data]

    def __sample__(self, idx):
        img = Image.open(self.data[idx]).convert('RGB')
        label = np.asarray([self.targets[idx]]).astype('float32')
        return img, label


def get_agedb(args, alg, name=None, num_labels=1000, num_classes=1, data_dir='./data', include_lb_to_ulb=True):

    data_dir = os.path.join(data_dir, 'agedb')
    df = pd.read_csv(os.path.join(data_dir, "agedb.csv"))
    df_train, df_val, df_test = df[df['split'] == 'train'], df[df['split'] == 'val'], df[df['split'] == 'test']
    train_labels, train_data = df_train['age'].tolist(), df_train['path'].tolist()
    test_labels, test_data = df_test['age'].tolist(), df_test['path'].tolist()
    # print(df_train['age'].shape, df_test['age'].shape)  # (12208,) (2140,)

    imgnet_mean = (0.485, 0.456, 0.406)
    imgnet_std = (0.229, 0.224, 0.225)
    img_size = args.img_size
    crop_ratio = args.crop_ratio

    transform_weak = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomCrop(img_size, padding=16, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(imgnet_mean, imgnet_std),
    ])

    transform_strong = transforms.Compose([
        transforms.Resize(int(math.floor(img_size / crop_ratio))),
        RandomResizedCropAndInterpolation((img_size, img_size), scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 10),
        transforms.ToTensor(),
        transforms.Normalize(imgnet_mean, imgnet_std)
    ])

    transform_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(imgnet_mean, imgnet_std),
    ])

    lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(args, train_data, train_labels, num_classes=1, 
                                                                lb_num_labels=num_labels,
                                                                ulb_num_labels=args.ulb_num_labels,
                                                                lb_imbalance_ratio=args.lb_imb_ratio,
                                                                ulb_imbalance_ratio=args.ulb_imb_ratio,
                                                                include_lb_to_ulb=include_lb_to_ulb)

    if alg == 'fullysupervised':
        lb_data = train_data
        lb_targets = train_labels

    lb_dset = AgeDBIDataset(alg, lb_data, lb_targets, num_classes,
                            transform_weak, False, None, False, data_dir=data_dir)
    ulb_dset = AgeDBIDataset(alg, ulb_data, ulb_targets, num_classes,
                            transform_weak, True, transform_strong, False, data_dir=data_dir)
    eval_dset = AgeDBIDataset(alg, test_data, test_labels, num_classes,
                            transform_val, False, None, False, data_dir=data_dir)

    return lb_dset, ulb_dset, eval_dset
