import torch
import torch.nn as nn
import torch.nn.functional as F

class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : [b, T, input_size]
        output : [b, T, output_size]
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)  # b x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # b x T x output_size
        return output
