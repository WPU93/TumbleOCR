import os
import sys
from easydict import EasyDict


__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))
# torch
import torch
from torch import nn
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# data
from tumbocr.data.load_rec_data import recDataset
from tumbocr.data.data_utils import get_vocabulary
from tumbocr.models.create_model import create_model
from tumbocr.optim_factory import create_optimizer
from tumbocr.scheduler_factory import create_scheduler
# utils
from tumbocr.utils.utils import from_pretrained,load_config
from tumbocr.utils.samplers.dataSampler import recSampler
from tumbocr.utils.transforms.get_transforms import get_transforms
from tumbocr.utils.data_parallel import BalancedDataParallel
#train&val
from trainval import train,validate

#_______________
from tumbocr.models.efficient import ViT
from tumbocr.models.linformer import Linformer
#_______________
def main(cfg):
    #---env_init---
    ngpus_per_node = torch.cuda.device_count()
    print("=> using {} GPU's".format(ngpus_per_node))
    char2id,id2char = get_vocabulary(cfg.Global.dict_path)
    cfg.Global.num_classes = len(char2id)
    cfg.Global.device = "cuda" if ngpus_per_node > 0 else "cpu"
    #---model---
    model = create_model(cfg.Model.arch)(cfg)
    # if cfg.Global.loss == "ctc":
    #     criterion_ctc = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    # elif cfg.Global.loss == "attn":
    #     criterion_att = nn.NLLLoss(reduction='none')
    # else:
    criterion_ctc = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    criterion_att = nn.NLLLoss(reduction='none')

    if cfg.Global.pretrained != "":
        print("=> using pre-trained model")
        if not os.path.isfile(cfg.Global.pretrained):
            print("=> no checkpoint found at '{}'".format(cfg.Global.pretrained))
        else:
            print("=> weights restore from '{}'".format(cfg.Global.pretrained))
            model = from_pretrained(model,cfg.Global.pretrained,is_frozen=False)
    if cfg.Global.balanced_bsz > 0:
        gpu0_bsz = cfg.Global.balanced_bsz
        model = BalancedDataParallel(gpu0_bsz,model,dim=0).cuda()
    else:
        model = torch.nn.DataParallel(model).to(cfg.Global.device)

    #---data---
    imgH,imgW = cfg.Train.image_shape[0],cfg.Train.image_shape[1]
    train_transform,val_transform = get_transforms(rand_aug=cfg.Train.rand_aug)
    train_dataset = recDataset(cfg.Train.train_path,cfg.Global.dict_path,
                    cfg.Global.out_seq_len,imgH,imgW,
                    train_transform,is_training=True,text_aug=cfg.Train.text_aug)
    val_dataset = recDataset(cfg.Val.val_path,cfg.Global.dict_path,
                  cfg.Global.out_seq_len,imgH,imgW,
                  val_transform,is_training=False,text_aug=False)
    
    train_sampler = recSampler(train_dataset,pow=cfg.Train.balanced_pow) if cfg.Train.balanced_pow > 0  else None
    print("train_sampler:",train_sampler)
    train_loader = DataLoader(dataset=train_dataset,
                                    batch_size=cfg.Train.batch_size,
                                    shuffle=(train_sampler is None),
                                    drop_last=True,
                                    sampler=train_sampler,
                                    num_workers=cfg.Train.workers)
    
    val_loader = DataLoader(dataset=val_dataset,
                                    batch_size=cfg.Val.batch_size,
                                    shuffle=False,
                                    drop_last=True,
                                    num_workers=cfg.Val.workers)

    cfg.Train.traindata_size = len(train_dataset)
    cfg.Train.num_batches = len(train_loader)

    #---optim & lr_scheduler---
    cfg.Optimizer.init_lr = cfg.Optimizer.lr
    cfg.Optimizer.lr = cfg.Optimizer.warmup_lr if (cfg.Optimizer.warmup > 0) else cfg.Optimizer.lr
    optimizer = create_optimizer(model,cfg)
    scheduler = create_scheduler(optimizer,cfg)
    print("=> optimizer",optimizer)
    if cfg.Global.resume == True:
        resume_checkpoint = torch.load(cfg.Global.pretrained)
        cfg.Global.start_epoch = resume_checkpoint['epoch']
        optimizer.load_state_dict(resume_checkpoint['optimizer'])
    #---epoch---
    best_acc = 0
    for epoch in range(cfg.Global.start_epoch,cfg.Global.epochs):
        # seq_acc,char_acc = validate(val_loader, model, epoch,id2char, cfg)
        train(train_loader, model, criterion_ctc, criterion_att, optimizer, epoch, scheduler, cfg)
        seq_acc,char_acc = validate(val_loader, model, epoch,id2char, cfg)
        if scheduler is not None and cfg.Optimizer.scheduler.name == "step":
            scheduler.step()
        is_best = char_acc >= best_acc
        best_acc = max(char_acc,best_acc)
        if is_best:
            best_path = "{}/{}_{}_{:.3f}_{:.3f}_best_model.pth".format(cfg.Global.checkpoints,cfg.Global.model_name,epoch,seq_acc,char_acc)
            print("=> {} epoch is best\t Best char_acc is {:.3f}\t Saving model to {}!".format(epoch,best_acc,best_path))
            state = {'state': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, best_path)

if __name__ == '__main__':
    cfg = load_config(sys.argv[1])
    cfg = EasyDict(cfg)
    main(cfg)
