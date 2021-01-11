import torch.nn as nn
import math
import torch
import torch.nn.functional as F


class encoder(nn.Module):

    def __init__(self, config):
        super(encoder, self).__init__()
        self.maxpool1 = nn.MaxPool2d((6, 1), stride=(6, 1))
        self.feat_channels = config.Model.backbone.feat_channels
        self.hidden_dim = config.Model.encoder.hidden_dim
        self.out_channels = self.hidden_dim
        self.encoder_name = config.Model.encoder.name
        if config.Model.encoder.name == "rnn":
            self.encode_lstm = nn.LSTM(
                self.feat_channels, self.hidden_dim, 1, bidirectional=True)
            self.linear_encode = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        elif config.Model.encoder.name == "fc":
            self.linear_encode = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        
    def forward(self,conv_f):
        
        x = self.maxpool1(conv_f)
        x = torch.squeeze(x, 2)  
        x = x.permute(2, 0, 1)
        if self.encoder_name == "rnn":
            self.encode_lstm.flatten_parameters()
            x, _ = self.encode_lstm(x)
            hidden_en = self.linear_encode(x)
        elif self.encoder_name == "fc":
            hidden_en = self.linear_encode(x)
        return hidden_en
        