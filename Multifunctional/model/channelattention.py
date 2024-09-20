# -*- coding: utf-8 -*-
import torch
from torch import nn

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):   
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)   
        self.max_pool = nn.AdaptiveMaxPool2d(1)   
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.4),
            nn.Linear(channel // reduction, channel, bias=False)  
        )

    def forward(self, x):   
        b, c, _, _ = x.size()
        y_avg = self.avg_pool(x).view(b, c)  
        y_max = self.max_pool(x).view(b, c)  
        y_avg = self.fc(y_avg).view(b, c, 1, 1) 
        y_max = self.fc(y_max).view(b, c, 1, 1) 
        y = nn.Sigmoid()(y_avg+y_max)  
        return x * y.expand_as(x)       



if __name__ == '__main__':
    a = torch.randn((20, 32, 16, 16))
    in_channel = a.shape[1]
    # print(a.shape)
    model = ChannelAttention(32,16,3)

    b = model(a)
    print(b)
    print(b.shape)