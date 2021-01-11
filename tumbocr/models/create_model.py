import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from .sar_recog import Recogniton
from .modules.recSequenceModel.Recognition import myRecogniton
from .modules.recSequenceModel.decoder import decoder
from .modules.recSequenceModel.encoder import encoder
from .crnn_recog import BidirectionalLSTM
from .modules.featureExtractor.rec_resnet import ResNet
from .modules.featureExtractor.rec_mobilenetv3 import MobileNetV3
from .modules.Prediction.Prediction_FC import Prediction_FC
from .modules.Prediction.Prediction_GCN import Prediction_GCN
from .modules.TPS.TPS import TPS_SpatialTransformerNetwork

class CRNN(nn.Module):

    def __init__(self, config):
        super(CRNN, self).__init__()
        self.featureExtractor = ResNet(31)
        self.hidden_dim = config.Model.hidden_dim
        self.num_classes = config.Global.num_classes
        self.featrue_layers = config.Model.featrue_layers
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))

        self.recg = nn.Sequential(
            BidirectionalLSTM(self.featrue_layers, self.hidden_dim, self.hidden_dim),
            BidirectionalLSTM(self.hidden_dim, self.hidden_dim, self.num_classes))

    def forward(self, image):

        feature = self.featureExtractor(image)#[b, c, h, w]
        feature = self.AdaptiveAvgPool(feature.permute(0,3,1,2))  # [b, c, h, w] -> [b,w,c,h]
        feature = feature.squeeze(3)# [w, b, c]
        out = self.recg(feature)
        out =  F.log_softmax(out,dim=2)
        if not self.training:
            idxscore, idx = torch.max(out, 2)
            return idx
        return out

class SAR(nn.Module):

    def __init__(self, config):
        super(SAR, self).__init__()
        self.featureExtractor = ResNet(31)
        self.recg = Recogniton(config)

    def forward(self, image, target=None):
        feature = self.featureExtractor(image)
        if self.training:
            out = self.recg(feature, target)
        else:
            seq = self.recg(feature)  # 2 32 | 2 32
            return seq
        return out

class recurAttnModel(nn.Module):

    def __init__(self, config):
        super(recurAttnModel, self).__init__()
        self.featureExtractor = ResNet(31)
        self.recg = myRecogniton(config)
        self.Prediction = nn.Linear(1024, config.Global.num_classes)

    def forward(self, image, target=None):
        feature = self.featureExtractor(image)
        sequenceout = self.recg(feature)
        out = self.Prediction(sequenceout)
        out =  F.log_softmax(out,dim=2)
        if not self.training:
            idxscore, idx = torch.max(out, 2)
            return idx
        return out

class myModel(nn.Module):

    def __init__(self, config):
        super(myModel, self).__init__()
        self.tps = None
        self.loss_type = config.Global.loss
        if config.Model.tps.name != None:
            self.tps = TPS_SpatialTransformerNetwork(config.Model.tps.num_fiducial,
                I_size=config.Train.image_shape, I_r_size=config.Train.image_shape, I_channel_num=3,name=config.Model.tps.name)
        self.featureExtractor = globals()[config.Model.backbone.name]()
        self.encoder = encoder(config)
        self.decoder = decoder(config)
        self.Prediction = Prediction_FC(self.decoder.out_channels, config.Global.num_classes)
        self.Prediction_attn= Prediction_FC(self.decoder.out_channels, config.Global.num_classes)
    def forward(self, image, target=None):
        if self.tps != None:
            image = self.tps(image)
        feature = self.featureExtractor(image)
        hidden_en = self.encoder(feature)
        ctc_out,attn_out = self.decoder(hidden_en,feature,target=target)
        if self.loss_type == "ctc":
            ctc_out = self.Prediction(ctc_out)
            ctc_out =  F.log_softmax(ctc_out,dim=2)
            if not self.training:
                idxscore, idx = torch.max(ctc_out, 2)
                return idx
            return ctc_out
        elif self.loss_type == "attn":
            attn_out = self.Prediction_attn(attn_out)
            attn_out = F.log_softmax(attn_out,dim=2)
            if not self.training:
                idxscore, idx = torch.max(attn_out, 2)
                return idx[:,1:]
            return attn_out
        else:
            ctc_out = self.Prediction(ctc_out)
            attn_out = self.Prediction_attn(attn_out)
            ctc_out =  F.log_softmax(ctc_out,dim=2)
            attn_out = F.log_softmax(attn_out,dim=2)
            if not self.training:
                idxscore, idx = torch.max(ctc_out+attn_out, 2)
                return idx[:,1:]
            return ctc_out,attn_out


def create_model(arch_name):
    return globals()[arch_name]