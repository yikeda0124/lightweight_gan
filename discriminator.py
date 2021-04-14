import math
import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt

class DownSampleConv(nn.Module):
    
    def __init__(self, chan_in, chan_out):
        super().__init__()
        
        self.downsample_conv = nn.Sequential(
            nn.Conv2d(chan_in, chan_out, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(chan_out),
            nn.Conv2d(chan_out, chan_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(chan_out),
            nn.LeakyReLU(0.2)
        )
        
        self.averagepool_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(2),
            nn.Conv2d(chan_in, chan_out, kernel_size=1),
            nn.BatchNorm2d(chan_out),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, input_img):
        return self.downsample_conv(input_img) + self.averagepool_conv(input_img)
    
    
class SimpleDecoder(nn.Module):
    
    def __init__(self, input_chan):
        super().__init__()
        
        def decoderBlock(chan_in, chan_out):
            decoder_block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(chan_in, chan_out*2, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(chan_out*2),
                nn.GLU(dim=1)
            )
        
        self.simple_decoder = nn.Sequential(
            decoderBlock(input_chan, input_chan//2),
            decoderBlock(input_chan//2, input_chan//4),
            decoderBlock(input_chan//4, input_chan//8),
            decoderBlock(input_chan//8, 3),
        )
    
    def forward(self, input_img):
        return self.simple_decoder(input_img)

class Discriminator(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        chan_in, chan_mid, chan_out = 3, 8, 16
        
        self.initial_conv = nn.Sequential(
            nn.Conv2d(chan_in, chan_mid, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(chan_mid, chan_out, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(chan_out),
            nn.LeakyReLU(0.2)
        )
        
        self.from256to128 = DownSampleConv(16, 32)
        self.from128to64 = DownSampleConv(32, 64)
        self.from64to32 = DownSampleConv(64, 128)
        self.from32to16 = DownSampleConv(128, 256)
        self.from16to8 = DownSampleConv(256, 512)
        
        self.decoder1 = SimpleDecoder(256)
        self.decoder2 = SimpleDecoder(512)
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 1, kernel_size=4)
        )
    
    def forward(self, input_img):
        img256 = self.inital_conv(input_img)
        
        img128 = self.from256to128(img256)
        img64 = self.from128to64(img128)
        img32 = self.from64to32(img64)
        img16 = self.from32to16(img32)
        img8 = self.from16to8(img16)
        
        cropimg8 = crop(img16)
        decoded_img1 = self.decoder1(cropimg8)
        decoded_img2 = self.decoder2(img8)
        
        logits = self.final_conv(img8)

if __name__ == '__main__':
    pass