# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import random
import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from semilearn.datasets.cv_datasets.datasetbase import BasicDataset
from semilearn.datasets.augmentation import RandomResizedCropAndInterpolation
from semilearn.datasets.utils import split_ssl_data


class RCFMNISTDataset(BasicDataset):

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
        super(RCFMNISTDataset, self).__init__(alg=alg, data=data, targets=targets, num_classes=num_classes,
                transform=transform, is_ulb=is_ulb, strong_transform=strong_transform, onehot=onehot, *args, **kwargs)

    def __sample__(self, idx):
        img = Image.fromarray(self.data[idx, ...]).convert('RGB')
        label = np.asarray([self.targets[idx]]).astype('float32')
        return img, label


def img_torch2numpy(img): # N C H W --> N H W C
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if len(img.shape) == 4:
        return np.transpose(npimg, (0, 2, 3, 1))
    elif len(img.shape) == 3:
        return np.transpose(npimg, (1, 2, 0))

def img_numpy2torch(img):# N H W C --> N C H W  
    if len(img.shape) == 4:
        tmp_img = np.transpose(img, (0, 3, 1 ,2))
    else:
        tmp_img = np.transpose(img, (2, 0 ,1))
    tcimg = torch.tensor(tmp_img)
    tcimg = (tcimg - 0.5) * 2 # normalize
    return tcimg


def get_all_batches(loader):
    data_iter = iter(loader)
    steplen = len(loader)
    img_list = []
    label_list = []

    for step in range(steplen):
        images, labels = data_iter.next()
        img_list.append(images)
        label_list.append(labels)
    img_tensor = torch.cat(img_list, dim=0)
    label_tensor = torch.cat(label_list, dim=0)
    
    return img_tensor, label_tensor


def rotate_img(img, degree=None):
    rotate_class = [(360.0 / 60) * i for i in range(60)]
    # rotate a image by PIL
    img = img / 2 + 0.5     # unnormalize
    pil_img = transforms.ToPILImage()(img)
    if degree is None:
        degree = random.sample(rotate_class, 1)[0]
    r_img = pil_img.rotate(degree)
    r_img = transforms.ToTensor()(r_img)  # read float
    r_img = (r_img - 0.5) * 2.0  # normalize
    return r_img, degree


def get_rotate_imgs(imgs):
    # rotate set of images
    r_img_list, degree_list = [], []
    for i in range(imgs.shape[0]):
        r_img, degree = rotate_img(imgs[i])
        r_img_list.append(r_img.unsqueeze(0))
        degree_list.append(torch.Tensor([int(degree)]))

    r_img_list = torch.cat(r_img_list, dim=0)
    degree_list = torch.cat(degree_list, dim=0)
    r_img_np = img_torch2numpy(r_img_list)
    degree_np = degree_list.numpy()

    return r_img_np, degree_list


def copydim(set:np.array, num=3):
    return np.expand_dims(set, -1).repeat(num, axis=-1)


def linear_red_blue_preparation(x_train, x_test, y_train, y_test, spurious_ratio=0.9):
    # # calculate the pearson between ratio and y
    x_train, ratio_reshape, y_reshape, idx2assay_train, assay2idx_train, _ = color_linear_red_blue(
        x_train, y_train, spurious_ratio, use_spurious=True, inv=False)
    x_test, ratio_reshape, y_reshape, _, _, test_assay2idx_list  = color_linear_red_blue(
        x_test, y_test, spurious_ratio, use_spurious=True, inv=False)

    return x_train, x_test, assay2idx_train, test_assay2idx_list


def color_linear_red_blue(x_set:np.array, y_set:np.array, spurious_ratio=0.9, use_spurious=1, inv=False):
    y_reshape = y_set
    x_tmp = x_set
    ratio_reshape = np.zeros_like(y_reshape)
    print(f'x_tmp.shape = {x_tmp.shape}, y_reshape.shape = {y_reshape.shape}')

    num = int(y_reshape.shape[0])
    idx = np.arange(num)
    idx2assay = np.zeros(num)

    if use_spurious: # spurious
        mixtype_normal_idx = np.random.choice(idx, size=int(num * spurious_ratio), replace=False)
        mixtype_inverse_idx = np.setdiff1d(idx, mixtype_normal_idx)
        ratio_matric = copydim(copydim(y_reshape, 1), 1)

        if inv == False: # normal spurious correlation
            x_tmp[mixtype_normal_idx] = red_blue_linear_map(x_tmp[mixtype_normal_idx], ratio_matric[mixtype_normal_idx])
            x_tmp[mixtype_inverse_idx] = red_blue_linear_map(x_tmp[mixtype_inverse_idx], 1.0 - ratio_matric[mixtype_inverse_idx])
            ratio_reshape[mixtype_normal_idx] = y_reshape[mixtype_normal_idx]
            ratio_reshape[mixtype_inverse_idx] = 1.0 - y_reshape[mixtype_inverse_idx]
        else: # inverse spurious correlation
            x_tmp[mixtype_normal_idx] = red_blue_linear_map(x_tmp[mixtype_normal_idx], 1.0 - ratio_matric[mixtype_normal_idx])
            x_tmp[mixtype_inverse_idx] = red_blue_linear_map(x_tmp[mixtype_inverse_idx], ratio_matric[mixtype_inverse_idx])
            ratio_reshape[mixtype_normal_idx] = 1.0 - y_reshape[mixtype_normal_idx]
            ratio_reshape[mixtype_inverse_idx] = y_reshape[mixtype_inverse_idx]

        idx2assay[mixtype_normal_idx] = 0 # class 0
        idx2assay[mixtype_inverse_idx] = 1 # class 1

    else: # test random
        ratio_reshape = np.random.rand(num)
        ratio = copydim(copydim(ratio_reshape, 1), 1)
        x_tmp = red_blue_linear_map(x_tmp, ratio)
        # all class 0
    x_set = x_tmp

    # ood
    assay2idx_list = [torch.nonzero(torch.tensor(idx2assay == loc)).squeeze(-1)
                                for loc in np.unique(idx2assay)] # ok
    assay2idx = {loc:torch.nonzero(torch.tensor(idx2assay == loc)).squeeze(-1)
                            for loc in np.unique(idx2assay)}
    return x_set, ratio_reshape, y_reshape, idx2assay, assay2idx, assay2idx_list


def red_blue_linear_map(imgt:np.array, red_ratio):
    color_lower_bound = 5 / 255  # 60 / 255    
    # R outside background -> red * ratio
    imgt[...,0] = np.where(imgt[...,0] > color_lower_bound , imgt[...,0] * red_ratio, imgt[...,0])
    # G outside background -> 0
    imgt[...,1] = np.where(imgt[...,1] > color_lower_bound , 0, imgt[...,1])
    # B outside background -> blue * (1 - ratio)
    imgt[...,2] = np.where(imgt[...,2] > color_lower_bound , imgt[...,2] * (1 - red_ratio), imgt[...,2])
    return imgt


def test_img(img, x, degree, name, iscolor = True):
    for i in range(3):
        img_save(img[i], f'{name}_{i}')
        if iscolor:
            img_save(x[i], f'{name}_{i}_color_r_{float(degree[i])}')
        else:
            img_save(x[i], f'{name}_{i}_no_color_r_{float(degree[i])}')


def img_save(img, file_name):
    
    plt.switch_backend('agg')
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])

    img = img / 2 + 0.5     # unnormalize
    if not isinstance(img, np.ndarray):
        npimg = img.numpy()
    else:
        npimg = img
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    
    plt.savefig(f'./{file_name}.jpg')


def get_rcfmnist(args, alg='', name=None, num_labels=1000, num_classes=1, data_dir='./data', include_lb_to_ulb=True):

    data_dir = os.path.join(data_dir, 'rcf_mnist')
    basic_data_transforms = transforms.Compose([
                                transforms.Resize(32),
                                transforms.Grayscale(3), 
                                transforms.ToTensor(), 
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                            ])
    train_set = FashionMNIST(root=data_dir, train=True, download=True, transform=basic_data_transforms)
    test_set = FashionMNIST(root=data_dir, train=False, download=True, transform=basic_data_transforms)
    train_loader = DataLoader(train_set, batch_size=1000, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False, num_workers=2)
    train_data_raw, _ = get_all_batches(train_loader)
    test_data_raw, _ = get_all_batches(test_loader)

    train_data, train_labels = get_rotate_imgs(train_data_raw)
    test_data, test_labels = get_rotate_imgs(test_data_raw)
    train_data = np.uint8(train_data)  # prepare for numpy to PIL
    test_data = np.uint8(test_data)
    train_labels, test_labels = train_labels.cpu().numpy(), test_labels.cpu().numpy()

    img_size = args.img_size
    crop_ratio = args.crop_ratio

    transform_weak = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomCrop(img_size, padding=2),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    transform_strong = transforms.Compose([
        transforms.Resize(int(math.floor(img_size / crop_ratio))),
        RandomResizedCropAndInterpolation((img_size, img_size), scale=(0.8, 1.)),
        transforms.RandomApply(torch.nn.ModuleList([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        ]), p=0.8),
        transforms.GaussianBlur(kernel_size=11),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
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

    lb_dset = RCFMNISTDataset(alg, lb_data, lb_targets, num_classes,
                            transform_weak, False, None, False)
    ulb_dset = RCFMNISTDataset(alg, ulb_data, ulb_targets, num_classes,
                            transform_weak, True, transform_strong, False)
    eval_dset = RCFMNISTDataset(alg, test_data, test_labels, num_classes,
                            transform_val, False, None, False)

    return lb_dset, ulb_dset, eval_dset


if __name__ == '__main__':
    get_rcfmnist(args=None)
