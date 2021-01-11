# TumbleOCR

## 简介
基于PyTorch实现中英文自然场景文字的识别工具（TumbleOCR），包括训练、推理、部署以及轻量化等多项功能。

## 实现功能
- [x]  支持单机多卡
- [x]  支持mobilenetv3和resnet
- [x]  添加了TPS模块
- [x]  添加dataSampler
- [x]  支持基于CRNN+CTC的实现
- [x]  支持基于Attention的[Show,Attent,Read](https://arxiv.org/pdf/1811.00751.pdf)的实现
- [x]  支持基于Show,Attent,Read的2d-attention+CTC的实现
- [x]  支持randaug和[textaug](https://arxiv.org/abs/2003.06606)两种ocr数据增强方式


## 环境部署
```
    torch==1.6.0
    torchvision==0.7.0
    opencv-python
    python-Levenshtein
    easydict
    pyyaml
运行./env.sh
```
## 模型训练
#### 生成相应的字典
- 包括英文识别字典，共计94个字符
- 包括中英文识别常用字典，共计6624个字符
#### 配置相应的config文件
通用的server模型[config](configs/rec_sar_train_config.yaml)
通用的mobile模型[config](configs/rec_ctc_mbv3_att2d_config.yaml)
```
运行./env.sh
```
## 数据增强

支持randaug和textaug两种数据增强

## 模型轻量化