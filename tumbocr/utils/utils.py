import os
import torch
import yaml
import torch.nn as nn
import Levenshtein

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:                              #全角空格直接转换            
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374): #全角字符（除空格）根据关系转化
            inside_code -= 65248
            rstring += chr(inside_code)
        else:
            rstring += chr(inside_code)
    return rstring


def seq_accurate(predict,target):
    assert len(predict) == len(target)
    count = 0
    for i in range(len(target)):
        if predict[i] == target[i]:
            count = count + 1
        else:
            continue
    return count/len(target)

def char_accurate(predict,target):
    ratio_sum = 0
    assert len(predict) == len(target)
    for i in range(len(target)):
        ratio = (Levenshtein.ratio(target[i], predict[i]))
        ratio_sum = ratio_sum + ratio
    char_acc = ratio_sum/len(target)
    return char_acc

def idx2str_ctc(idx,id2char,PAD=0,UNK=6624,EOS=6625,SOS=6625):
    text = ""
    for i in range(len(idx)):
        if idx[i]!=PAD and idx[i]!=UNK and (not (i>0 and idx[i-1]==idx[i])): # PAD & UNK
            text += id2char[idx[i]]
    return text

def idx2str_attn(idx,id2char,PAD=0,UNK=6624,EOS=6625,SOS=6625):
    text = ""
    for i in range(len(idx)):
        if idx[i]==PAD or idx[i]==EOS:# EOS & PAD
            break
        elif idx[i]==UNK:# UNK
            continue
        else: 
            text += id2char[idx[i]]
    return  text

def from_pretrained(model,weights_path,is_frozen=False):
    pretrained_dict = torch.load(weights_path,map_location=torch.device('cpu'))["state"]
    new_pretrained_dict = {}
    model_dict = model.state_dict()
    for i, (k, v) in enumerate(pretrained_dict.items()):
        new_k = k[7:] if k.startswith('module') else k
        if new_k in model_dict:
            new_pretrained_dict[new_k]=v
        else:
            print(new_k)
    model_dict.update(new_pretrained_dict)
    model.load_state_dict(model_dict)
    if is_frozen:
        for name,param in model.named_parameters():
            if "featureExtractor" in name:
                param.requires_grad = False
            else:
                print("=> {} is not frozen".format(name))
    return model

def load_config(file_path):
    ex = os.path.splitext(file_path)[1]
    assert ex in ['.yml', '.yaml']
    with open(file_path, 'rb') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    return config



