import torch
from torch import nn

class Prediction_FC(nn.Module):
    def __init__(self, in_channels,num_classes):
        super().__init__()
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        return self.fc(x)