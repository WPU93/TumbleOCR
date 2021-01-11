import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

import time
import torch
from torch import nn
import torch.utils.data as data
import torchvision.transforms as transforms
from tumbocr.utils.utils import AverageMeter
from tumbocr.utils.utils import seq_accurate,char_accurate
from tumbocr.utils.utils import idx2str_ctc,idx2str_attn

def train(train_loader, model, criterion_ctc, criterion_att,optimizer,epoch,scheduler, config):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss = AverageMeter()
    seq_acc = AverageMeter()
    char_acc = AverageMeter()
    model.train()

    for batch_idx, (imgs,targets,length,text) in enumerate(train_loader):
        st = time.time()
        imgs = imgs.cuda(config.Global.gpu, non_blocking=True)
        targets = targets.cuda(config.Global.gpu, non_blocking=True)
        length = length.cuda(config.Global.gpu, non_blocking=True)
        data_time.update(time.time() - st)
        if config.Global.loss == "ctc":
            preds = model(imgs)# b T C
            preds = preds.permute(1, 0, 2)# b,T,C -> T,b,C
            preds_length = torch.IntTensor([preds.size(0)] * config.Train.batch_size).cuda(config.Global.gpu, non_blocking=True)
            losses = criterion_ctc(preds, targets, preds_length, length)
        elif config.Global.loss == "attn" :
            l_target = torch.zeros(config.Train.batch_size,config.Global.out_seq_len,
                   dtype=torch.long).cuda(config.Global.gpu, non_blocking=True)
            l_target[:, 1:] = targets[:,:-1]
            out = model(imgs, targets)
            out = out.view(-1,config.Global.num_classes)
            l_target = l_target.view(-1)
            mask = torch.eq(l_target, 0)
            losses = criterion_att(out,l_target)[~mask]
            losses = torch.sum(losses)/config.Train.batch_size
        else:
            ctc_out, attn_out = model(imgs, l_target)
            preds = preds.permute(1, 0, 2)# b,T,C -> T,b,C
            preds_length = torch.IntTensor([preds.size(0)] * config.Train.batch_size).cuda(config.Global.gpu, non_blocking=True)
            ctc_losses = criterion_ctc(preds, targets, preds_length, length)

            l_target = torch.zeros(config.Train.batch_size,config.Global.out_seq_len,
                   dtype=torch.long).cuda(config.Global.gpu, non_blocking=True)
            l_target[:, 1:-1] = targets
            l_target = l_target.view(-1)
            mask = torch.eq(l_target, 0)
            attn_losses = criterion_att(attn_out,l_target)[~mask]
            attn_losses = torch.mean(attn_losses)
            losses = ctc_losses + attn_losses

        loss_avg = losses.mean()
        lr=optimizer.param_groups[0]['lr']
        optimizer.zero_grad()
        # loss_avg.backward(retain_graph=True)
        loss_avg.backward()
        optimizer.step()
        if scheduler is not None and config.Optimizer.scheduler.name == "cosine":
            scheduler.step()
        loss.update(loss_avg.item())
        batch_time.update(time.time() - st)
        if batch_idx % config.Global.print_feq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Lr: {lr:.6f}'.format(epoch, batch_idx, len(train_loader),
                   batch_time = batch_time,data_time=data_time,loss=loss,lr=lr))
        if (batch_idx+1) % config.Global.save_iter == 0:
            save_path = config.Global.checkpoints + config.Global.model_name + "-" + str(epoch) + '-' +str(batch_idx) + '.pth'
            print("=> Saving model to {}".format(save_path))
            state = {'state': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'batch_idx': batch_idx}
            torch.save(state, config.Global.checkpoints + config.Global.model_name + "-" + str(epoch) + '-' +str(batch_idx) + '.pth')
        

def validate(val_loader, model, epoch,id2char, config):
    model.eval()
    pred_list = []
    target_list = []
    batch_pred_time = AverageMeter()
    for batch_idx, (imgs,targets,length,text) in enumerate(val_loader):
        imgs = imgs.cuda(config.Global.gpu, non_blocking=True)
        targets = targets.cuda(config.Global.gpu, non_blocking=True)
        st = time.time()
        if config.Global.loss == "ctc":
            pred_tensor = model(imgs)
            preds = pred_tensor.cpu().detach().numpy()
        elif config.Global.loss == "attn":
            pred_tensor = model(imgs, targets)
            preds = pred_tensor.cpu().numpy()
    
        batch_pred_time.update(time.time() - st)
        targets = targets.cpu().numpy()
        for i in range(preds.shape[0]):
            text_target = text[i]
            if config.Global.loss == "ctc":
                text_pred = idx2str_ctc(preds[i],id2char)
            elif config.Global.loss == "attn":
                text_pred = idx2str_attn(preds[i],id2char)
            pred_list.append(text_pred)
            target_list.append(text_target)
        if (batch_idx+1) % config.Global.print_feq == 0:
            seq_acc = seq_accurate(pred_list,target_list)
            char_acc = char_accurate(pred_list,target_list)
            print('Epoch: [{}][{}/{}]\t'
                  'Batch_pred_time: {batch_pred_time.val:.3f} ({batch_pred_time.avg:.3f})\t'
                  'Seq_acc: {seq_acc:.4f}\t'
                  'Char_acc: {char_acc:.4f}'.format(epoch, batch_idx, len(val_loader),
                   batch_pred_time=batch_pred_time,seq_acc=seq_acc,char_acc=char_acc))
    seq_acc = seq_accurate(pred_list,target_list)
    char_acc = char_accurate(pred_list,target_list)
    print("ALL_seq_accurate:{:.4f}\t ALL_char_accurate:{:.4f}".format(seq_acc,char_acc))
    return seq_acc,char_acc
