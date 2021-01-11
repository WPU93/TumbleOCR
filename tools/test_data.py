import os
import argparse
from PIL import Image
import numpy as np
import cv2
import time
import sys
from easydict import EasyDict
# torch
import torch
from torch import nn
# data
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from data.load_data_url import ocrDataset
from data import load_data

if __name__=="__main__":
    image_transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
    ]
    )
    maxlen, imgH, imgW = 30, 48, 160
    data_path = "/data/remote/ocr_data/OCR_GT/90k_SynthText_SynthAdd.txt"
    dict_path = "dict/dict.txt"
    train_dataset = ocrDataset(data_path,dict_path,maxlen,imgH,imgW,image_transform)
    print("num of data:",len(train_dataset))
    # train_sampler = dataSampler(train_dataset,pow=0.5)
    trainset_dataloader = DataLoader(dataset=train_dataset,
                                     batch_size=1024,
                                     shuffle=False,
                                    #  sampler=train_sampler,
                                     num_workers=32)
    for i_batch, (image,label,length,text) in enumerate(trainset_dataloader):
        if i_batch%10 == 1:
            print(i_batch/len(trainset_dataloader))

