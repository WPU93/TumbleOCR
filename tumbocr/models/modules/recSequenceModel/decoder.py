import torch.nn as nn
import math
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

class decoder(nn.Module):

    def __init__(self, config):
        super(decoder, self).__init__()
        self.loss_type = config.Global.loss
        self.device = config.Global.device
        self.out_seq_len = config.Global.out_seq_len
        self.num_classes = config.Global.num_classes
        self.feat_channels = config.Model.backbone.feat_channels
        self.hidden_dim = config.Model.encoder.hidden_dim
        self.hidden_dim_de = config.Model.decoder.hidden_dim_de
        self.embedding_size = config.Model.decoder.embedding_size
        self.attn_embedding = config.Model.decoder.attn_embedding

        self.attn_channels = self.embedding_size if self.attn_embedding else self.feat_channels
        self.out_channels = self.hidden_dim_de + self.attn_channels
        self.conv_em = nn.Conv2d(self.feat_channels,self.embedding_size, kernel_size=3, padding=1)
        self.attention_nn = Attention_nn(self.hidden_dim_de, self.embedding_size)
        self.decoder_ctc = decoder_ctc(self.hidden_dim, self.hidden_dim_de)
        # self.decoder_attn = decoder_attn(self.hidden_dim, self.hidden_dim_de, self.attn_channels,
                                # self.out_seq_len, self.num_classes, self.device, num_layers=2)
    def forward(self, hidden_en ,conv_f, target=None):
        batch_size, feat_channels, feature_h, feature_w = conv_f.shape
        self.ctc_out = torch.zeros(batch_size, self.out_seq_len ,self.out_channels).to(self.device)
        x_em = self.conv_em(conv_f)
        att_x_em = x_em if self.attn_embedding else conv_f
        att_x_em = rearrange(att_x_em,'b c h w -> b (h w) c')

        if self.loss_type == "ctc":
            hidden_de = self.decoder_ctc(hidden_en)
            for t in range(len(hidden_de)):
                att, attw = self.attention_nn(hidden_de[t], x_em, att_x_em, feature_h, feature_w)
                self.ctc_out[:,t] = torch.cat((hidden_de[t], att), -1)
            return self.ctc_out
        # elif self.loss_type == "attn":
        #     attn_out = self.decoder_attn(hidden_en, x_em, att_x_em, feature_h, feature_w, target)
        #     return attn_out


            
class decoder_ctc(nn.Module):
    def __init__(self, hidden_dim, hidden_dim_de):
        super(decoder_ctc, self).__init__()
        self.hidden_dim = hidden_dim
        self.hidden_dim_de = hidden_dim_de
        self.decode_lstm = nn.LSTM(self.hidden_dim, self.hidden_dim_de, 1, bidirectional=True)
        self.linear_decode = nn.Linear(self.hidden_dim_de * 2, self.hidden_dim_de)

    def forward(self,hidden_en):
        hidden_de, _ = self.decode_lstm(hidden_en)
        hidden_de = self.linear_decode(hidden_de)
        return hidden_de

class Attention_nn(nn.Module):

    def __init__(self, hidden_dim_de, embedding_size):
        super(Attention_nn, self).__init__()
        self.hidden_dim_de = hidden_dim_de
        self.embedding_size = embedding_size

        self.h_em = nn.Conv2d(self.hidden_dim_de, self.embedding_size,kernel_size=1, stride=1)
        self.conv_att = nn.Conv2d(self.hidden_dim_de, 1,kernel_size=1, stride=1)
        self.dropout = nn.Dropout(p=0.5)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, h, x_em, att_x_em, feature_h, feature_w):
        h = rearrange(h,'b c -> b c 1 1')
        h_em = self.h_em(h)
        hemr = repeat(h_em,"b c 1 1 -> b c h w",h=feature_h, w=feature_w)
        feat = self.dropout(torch.tanh(x_em + h_em))
        feat_att = self.conv_att(feat)
        feat_att = rearrange(feat_att,'b c h w -> b (c h w)')
        alpha = self.softmax(feat_att)
        alpha = rearrange(alpha,'b hw -> b 1 hw')
        att_out = torch.matmul(alpha, att_x_em)
        att_out = rearrange(att_out,'b 1 dim -> b dim')
        return att_out, alpha



class decoder_attn(nn.Module):
    def __init__(self, hidden_dim, hidden_dim_de, attn_channels, out_seq_len, num_classes, device, num_layers=1):
        super(decoder_attn, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.hidden_dim_de = hidden_dim_de
        self.attn_channels = attn_channels
        self.out_seq_len = out_seq_len
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.decode_lstm = nn.ModuleList()
        for i in range(self.num_layers):
            self.decode_lstm.append(nn.LSTMCell(self.hidden_dim,self.hidden_dim_de))
        self.linear_attn = nn.Linear(self.hidden_dim_de+self.attn_channels, self.num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.attention_nn = Attention_nn(self.hidden_dim_de, self.attn_channels)
        self.embed = nn.Embedding(self.num_classes, self.hidden_dim)

    def forward(self, hidden_en, x_em, att_x_em, feature_h, feature_w, target=None):
        batch_size =  x_em.shape[0]
        self.hidden_de = []
        for i in range(self.num_layers):
            self.hidden_de.append((torch.zeros(batch_size, self.hidden_dim_de).to(self.device)
                , torch.zeros(batch_size, self.hidden_dim_de).to(self.device)))
        self.attn_out = torch.zeros(batch_size, self.out_seq_len, self.num_classes).to(self.device)
        self.seq = torch.zeros(batch_size, self.out_seq_len).to(self.device)
        if self.training:
            for t in range(self.out_seq_len):
                if t == 0:
                    xt = hidden_en[-1]
                else:
                    it = target[:, t - 1].view(-1).to(torch.long).to(self.device)
                    xt = self.embed(it)
                for i in range(self.num_layers):
                    inp = xt if i==0 else self.hidden_de[i - 1][0]               
                    self.hidden_de[i] = self.decode_lstm[i](inp, self.hidden_de[i])
                h = self.hidden_de[-1][0]
                att, attw = self.attention_nn(h, x_em, att_x_em, feature_h, feature_w)
                tmpcat = torch.cat((h, att), -1)
                scores = self.logsoftmax(self.linear_attn(tmpcat))
                self.attn_out[:, t, :] = scores
            return self.attn_out
        else:
            for t in range(self.out_seq_len):
                if t == 0:
                    xt = hidden_en[-1]
                elif t == 1:
                    it = torch.zeros(self.batch_size).to(torch.long).to(self.device)
                    xt = self.embed(it)
                else:
                    it = self.seq[:, t - 1]
                    it = it.view(-1).to(torch.long).to(self.device)
                    xt = self.embed(it)
                for i in range(self.num_layers):
                    inp = xt if i==0 else self.hidden_de[i - 1][0]                   
                    self.hidden_de[i] = self.decode_lstm[i](inp, self.hidden_de[i])
                h = self.hidden_de[-1][0]
                
                att, attw = self.attention_nn(h, x_em, att_x_em, feature_h, feature_w)
                tmpcat = torch.cat((h, att), -1)
                scores = self.logsoftmax(self.linear_attn(tmpcat))
                idxscore, idx = torch.max(scores, 1)
            return self.seq[:,1:]