##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import torch
import torchvision.transforms as transforms
from .autoaug import RandAugment
def get_transforms(rand_aug=True):

    normalize = transforms.Lambda(lambda img: img * 2.0 - 1.0)   
    if rand_aug:
        transform_train = transforms.Compose([
            RandAugment(2, 10),
            transforms.ToTensor(),
            normalize,
            ])
    else:
        transform_train = transforms.Compose([
            transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
            transforms.ToTensor(),
            normalize
            ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])

    
    return transform_train, transform_val

