import torch.nn as nn
import math
import torch
import torch.nn.functional as F


class myRecogniton(nn.Module):

    def __init__(self, config):
        super(myRecogniton, self).__init__()
        self.device = config.Global.device
        self.out_seq_len = config.Global.out_seq_len
        self.num_classes = config.Global.num_classes

        self.featrue_layers = config.Model.featrue_layers
        self.hidden_dim = config.Model.hidden_dim
        self.hidden_dim_de = config.Model.hidden_dim_de

        self.NULL_TOKEN = 0
        self.START_TOKEN = config.Global.num_classes - 1
        self.END_TOKEN = config.Global.num_classes - 1

        self.maxpool1 = nn.MaxPool2d((6, 1), stride=(6, 1))
        self.encode_lstm = nn.LSTM(
            self.featrue_layers, self.hidden_dim, 1, bidirectional=True)
        self.decode_lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim_de, 1, bidirectional=True)
        self.linear_encode = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.linear_decode = nn.Linear(self.hidden_dim_de * 2, self.hidden_dim_de)

        self.embed = nn.Embedding(self.num_classes, self.hidden_dim)

        self.attention_nn = Attention_nn(config)

    def forward(self, conv_f):
        # print("conv_f:",conv_f.shape) # 2, 512, 6, 40
        self.encode_lstm.flatten_parameters() 
        self.decode_lstm.flatten_parameters() 
        self.batch_size = conv_f.shape[0]
        self.out_seq_len = conv_f.shape[-1]
        self.out = torch.zeros(self.batch_size, self.out_seq_len ,self.hidden_dim_de + self.featrue_layers).to(self.device)
        # self.out = torch.zeros(self.batch_size, self.out_seq_len ,self.hidden_dim_de + self.featrue_layers)
        x = self.maxpool1(conv_f)
        
        x = torch.squeeze(x, 2)
        x = x.permute(2, 0, 1)
        # print(x.shape)#40, 2, 512
        x, _ = self.encode_lstm(x)
        # print(x.shape)#40, 2, 1024
        hidden_en = self.linear_encode(x)
        hidden_de, _ = self.decode_lstm(hidden_en)
        hidden_de = self.linear_decode(x)
        
        for i in range(len(hidden_de)):
            att, attw = self.attention_nn(conv_f, hidden_de[i])
            tmpcat = torch.cat((hidden_de[i], att), -1)
            self.out[:,i] = tmpcat
        return self.out
        

class Attention_nn(nn.Module):

    def __init__(self, config):
        super(Attention_nn, self).__init__()
        self.featrue_layers = config.Model.featrue_layers
        self.hidden_dim_de = config.Model.hidden_dim_de
        self.embedding_size = config.Model.embedding_size
        self.conv_h = nn.Linear(self.hidden_dim_de, self.embedding_size)
        self.conv_f = nn.Conv2d(self.featrue_layers,
                                self.embedding_size, kernel_size=3, padding=1)
        self.conv_att = nn.Linear(self.embedding_size, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, conv_f, h):  # [-1, 512, 6, 40]  | -1 512
        g_em = self.conv_h(h)
        g_em = g_em.view(g_em.shape[0], -1, 1)
        g_em = g_em.repeat(1, 1, conv_f.shape[
                           2] * conv_f.shape[3])  # -1 512 h*w
        g_em = g_em.permute(0, 2, 1)
        x_em = self.conv_f(conv_f)
        x_em = x_em.view(x_em.shape[0], -1, g_em.shape[1])
        x_em = x_em.permute(0, 2, 1)
        feat = self.dropout(torch.tanh(x_em + g_em))  # -1 h*w 512
        e = self.conv_att(feat.view(-1, self.embedding_size))  # -1*h*w 1
        alpha = self.softmax(e.view(-1,  g_em.shape[1]))  # -1  h*w
        alpha2 = alpha.view(-1, 1,  g_em.shape[1])  # -1 1 h*w
        orgfeat_embed = conv_f.view(-1, self.featrue_layers,
                                    g_em.shape[1])
        orgfeat_embed = orgfeat_embed.permute(0, 2, 1)  # -1 h*w 512
        att_out = torch.matmul(alpha2, orgfeat_embed)
        att_out = att_out.view(-1, self.featrue_layers)  # -1 512
        return att_out, alpha
