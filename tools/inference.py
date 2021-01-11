import os
import argparse
import sys 
import time
# torch
import torch
from torch import nn
# data
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from data.load_data import ocrDataset
from data.data_utils import get_vocabulary
# model
from models.create_model import SAR,CRNN,recurAttnModel,recurAttnModel_v2,DeAttnCTCModel
from trainval import train,validate
# utils
from config import get_args
from utils.utils import seq_accurate,char_accurate
from utils.utils import idx2str_crnn,idx2str_sar
from utils.utils import from_pretrained,load_config
from easydict import EasyDict

def main(args):
    cfg = load_config("configs/rec_recurAttnModel_train_config.yaml")
    cfg = EasyDict(cfg)
    ngpus_per_node = torch.cuda.device_count()
    char2id,id2char = get_vocabulary(cfg.Global.dict_path)

    cfg.Global.device = "cuda" if ngpus_per_node > 0 else "cpu"
    cfg.Global.num_classes = len(char2id)
    pretrained_path = "../save_models/recurAttnModel-scrach_11_0.722_0.845_best_model.pth"
    # pretrained_path = "save_models/real-mysynth-all-sum-balanced-256-6-10000.pth"
    cfg.Model.arch = "crnn" if "crnn" in pretrained_path else "recurAttnModel"
    if cfg.Model.arch == "crnn":
        model = CRNN(args).cuda() 
    elif cfg.Model.arch == "sar":
        model = SAR(args).cuda() 
    else:
        model = DeAttnCTCModel(cfg).cuda()
    inp = torch.rand([1,3,48,160]).cuda()
    out = model(inp)
    print(out.shpae)
if __name__ == '__main__':
    args =get_args(sys.argv[1:])
    main(args)
