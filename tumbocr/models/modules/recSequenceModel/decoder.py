import torch.nn as nn
import math
import torch
import torch.nn.functional as F


class decoder(nn.Module):

    def __init__(self, config):
        super(decoder, self).__init__()
        # self.NULL_TOKEN = 0
        # self.START_TOKEN = config.Global.num_classes - 1
        # self.END_TOKEN = config.Global.num_classes - 1
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
        self.decode_lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim_de, 1, bidirectional=True)
        self.linear_decode = nn.Linear(self.hidden_dim_de * 2, self.hidden_dim_de)

        self.attention_nn = Attention_nn(config)
        self.conv_em = nn.Conv2d(self.feat_channels,
                    self.embedding_size, kernel_size=3, padding=1)
        
        #--------------
        self.decode_lstm_att = nn.ModuleList()
        self.embed = nn.Embedding(self.num_classes, self.hidden_dim)
        for i in range(2):
            self.decode_lstm_att.append(nn.LSTMCell(self.hidden_dim, self.hidden_dim_de))
        #----------------

    def forward(self, hidden_en ,conv_f, target=None):
        #init 
        feature_h, feature_w = conv_f.shape[2], conv_f.shape[3]
        x_em = self.conv_em(conv_f) # -1,embedding_size,h,w
        att_x_em = x_em if self.attn_embedding else conv_f
        att_x_em = att_x_em.view(-1, self.attn_channels ,feature_h * feature_w)
        att_x_em = att_x_em.permute(0, 2, 1)# -1 240 embedding_size
        self.batch_size = conv_f.shape[0]
        self.out_seq_len = conv_f.shape[-1]
        self.ctc_out = torch.zeros(self.batch_size, self.out_seq_len ,self.out_channels).to(self.device)

        hidden_de, _ = self.decode_lstm(hidden_en)
        hidden_de = self.linear_decode(hidden_de)

        for t in range(len(hidden_de)):
            att, attw = self.attention_nn(hidden_de[t], x_em, att_x_em, feature_h, feature_w)
            self.ctc_out[:,t] = torch.cat((hidden_de[t], att), -1)
        if self.loss_type == "ctc":
            return self.ctc_out, _
        #——————————————————————————————————
        self.attn_out = torch.zeros(self.batch_size, self.out_seq_len ,self.out_channels).to(self.device)
        self.hidden_de_att = []
        for i in range(2):
            self.hidden_de_att.append((torch.zeros(self.batch_size, self.hidden_dim_de).to(self.device)
                , torch.zeros(self.batch_size, self.hidden_dim_de).to(self.device)))
        for t in range(len(hidden_de)):
            if t == 0:
                xt = hidden_en[-1]
            else:
                if self.training:
                    it = target[:, t - 1].view(-1).to(torch.long).to(self.device)
                else:
                    it = torch.zeros(self.batch_size).fill_(0).to(torch.long).to(self.device)
                xt = self.embed(it)
                
            for i in range(2):
                if i == 0:
                    inp = xt
                else:
                    inp = self.hidden_de_att[i - 1][0]
                self.hidden_de_att[i] = self.decode_lstm_att[i](inp, self.hidden_de_att[i])
            att, attw = self.attention_nn(self.hidden_de_att[-1][0], x_em, att_x_em, feature_h, feature_w)
            
            self.attn_out[:, t, :] = torch.cat((self.hidden_de_att[-1][0], att), -1)
        if self.loss_type == "attn":
            return _, self.attn_out
        #——————————————————————————————————
        return self.ctc_out,self.attn_out
        

class Attention_nn(nn.Module):

    def __init__(self, config):
        super(Attention_nn, self).__init__()
        self.hidden_dim_de = config.Model.decoder.hidden_dim_de
        self.embedding_size = config.Model.decoder.embedding_size

        self.conv_h = nn.Conv2d(self.hidden_dim_de, self.embedding_size,kernel_size=1, stride=1)
        self.conv_att = nn.Conv2d(self.embedding_size, 1,kernel_size=1, stride=1)
        self.dropout = nn.Dropout(p=0.5)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, h, x_em, att_x_em, feature_h, feature_w):
        h = h.reshape(-1,h.shape[1],1,1)
        g_em = self.conv_h(h)
        g_em = g_em.repeat(1, 1, feature_h, feature_w)
        feat = self.dropout(torch.tanh(x_em + g_em))
        feat_att = self.conv_att(feat)
        alpha = self.softmax(feat_att.view(-1,  feature_h*feature_w))
        alpha = alpha.view(-1, 1,  feature_h*feature_w)
        att_out = torch.matmul(alpha, att_x_em)
        att_out = att_out.view(-1, att_out.shape[2])# -1 attn_channels
        return att_out, alpha
