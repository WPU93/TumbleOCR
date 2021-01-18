import os
import sys 
import time
from easydict import EasyDict
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))
# torch
import torch
from torch import nn
# data
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from tumbocr.data.load_data import recDataset
from tumbocr.data.data_utils import get_vocabulary
# model
from tumbocr.models.create_model import create_model
from trainval import train,validate
# utils
from tumbocr.utils.utils import seq_accurate,char_accurate
from tumbocr.utils.utils import idx2str_ctc,idx2str_attn
from tumbocr.utils.utils import from_pretrained,load_config

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
def E_trans_to_C(string):
    E_pun = u':,.!?[]()<>"\''
    C_pun = u'：，。！？【】（）《》“‘'
    table= {ord(f):ord(t) for f,t in zip(E_pun,C_pun)}
    return string.translate(table)

def main():
    cfg = load_config("configs/rec_attn_ctc_mbv3_config.yaml")
    cfg = EasyDict(cfg)
    ngpus_per_node = torch.cuda.device_count()
    char2id,id2char = get_vocabulary(cfg.Global.dict_path)

    cfg.Global.device = "cuda" if ngpus_per_node > 0 else "cpu"
    cfg.Global.num_classes = len(char2id)
    # pretrained_path = "../save_models/myModel-MobileNetV3-1-9999.pth"
    pretrained_path = "../save_models/myModel-rec-cn-MobileNetV3L-attn-1-4999.pth"
    model = create_model(cfg.Model.arch)(cfg).cuda()
    model = from_pretrained(model,pretrained_path)
    model.eval()
    normalize = transforms.Lambda(lambda img: img * 2.0 - 1.0)
    image_transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        normalize,
    ])
    criterion = None    
    # load ckpt

    test_dataset = recDataset(cfg.Val.val_path,cfg.Global.dict_path,
              cfg.Global.out_seq_len,cfg.Val.image_shape[0],cfg.Val.image_shape[1],
              image_transform)
    test_loader = DataLoader(dataset=test_dataset,
                                    batch_size=10,
                                    shuffle=False,
                                    num_workers=4)

    for epoch in range(0,1):
        model.eval()
        pred_list = []
        target_list = []
        for batch_idx, (imgs,targets,length,text) in enumerate(test_loader):
            imgs = imgs.cuda(cfg.Global.gpu, non_blocking=True)
            targets = targets.cuda(cfg.Global.gpu, non_blocking=True)
            st = time.time()
            if cfg.Global.loss == "attn":
                pred_tensor = model(imgs, targets)
                preds = pred_tensor.cpu().numpy()
            if cfg.Global.loss == "ctc":
                pred_tensor = model(imgs)
                preds = pred_tensor.cpu().detach().numpy()
            cost = time.time()-st
            targets = targets.cpu().numpy()
            print(targets,preds)
            for i in range(preds.shape[0]):
                text_target = text[i]
                if cfg.Global.loss == "ctc":
                    text_pred = idx2str_ctc(preds[i],id2char)
                elif cfg.Global.loss == "attn":
                    text_pred = idx2str_attn(preds[i],id2char)
                # if E_trans_to_C(text_target)!=E_trans_to_C(text_pred):
                print("pred:{}\tlabel:{}\tcost:{}".format(text_pred,text_target,cost))
                # wf.write(text_target+"\t"+text_pred+"\n")
                pred_list.append(text_pred)
                target_list.append(text_target)
                 
        print(len(pred_list))
        seq_acc = seq_accurate(pred_list,target_list)
        char_acc = char_accurate(pred_list,target_list)
        print("seq_accurate:{:.4f},char_accurate:{:.4f}".format(seq_acc,char_acc))
        return seq_acc,char_acc

if __name__ == '__main__':
    main()
