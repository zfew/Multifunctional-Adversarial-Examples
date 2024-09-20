# -*- coding: utf-8 -*-
import torch
from torch import nn

from channelattention import ChannelAttention
import torchvision.models as models
from shuffleAttention import ShuffleAttention


class Encoder(nn.Module):
    def __init__(self, data_depth, hidden_size, color_band=3):  
        super(Encoder, self).__init__()
        self.data_depth = data_depth
        self.hidden_size = hidden_size
        self.color_band = color_band
        self.add_image = True
        self.build_models()

    def build_models(self):
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.color_band, out_channels=self.hidden_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_size),
            nn.LeakyReLU(inplace=True)
        )
        self.ca1 = ChannelAttention(self.hidden_size)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.hidden_size + self.data_depth, out_channels=self.hidden_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_size),
            nn.LeakyReLU(inplace=True)
        )
        self.ca2 = ChannelAttention(self.hidden_size)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.hidden_size*2 + self.data_depth, out_channels=self.hidden_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_size),
            nn.LeakyReLU(inplace=True)
        )
        self.ca3 = ChannelAttention(self.hidden_size)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=self.hidden_size*3 + self.data_depth, out_channels=self.color_band, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, image, message): 


        x = self.conv1(image)   
        x = self.ca1(x)    
        x_list = [x]   
        x = self.conv2(torch.cat(x_list + [message], dim=1)) 
        x = self.ca2(x)   
        x_list.append(x)   
        x = self.conv3(torch.cat(x_list + [message], dim=1)) 
        x = self.ca3(x)   
        x = self.conv4(torch.cat(x_list + [message], dim=1))  

        x = x.clamp(-0.1, 0.1)    
        if self.add_image:
            x = image + x   
        x.clamp_(0.0, 1.0)    

        return x




