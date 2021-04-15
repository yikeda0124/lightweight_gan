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
            nn.LeakyReLU(0.2),
            nn.Conv2d(chan_out, chan_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(chan_out),
            nn.LeakyReLU(0.2)
        )
        
        self.averagepool_conv = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(chan_in, chan_out, kernel_size=1),
            nn.BatchNorm2d(chan_out),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, input_img):
        return (self.downsample_conv(input_img) + self.averagepool_conv(input_img))/2
    
    
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
            return decoder_block
        
        self.simple_decoder = nn.Sequential(
            decoderBlock(input_chan, input_chan//2),
            decoderBlock(input_chan//2, input_chan//4),
            decoderBlock(input_chan//4, input_chan//8),
            decoderBlock(input_chan//8, 3)
        )
    
    def forward(self, input_img):
        return self.simple_decoder(input_img)
    
def crop(input_big, input_sml, l_size, s_size):
    img_size = input_sml.shape[3]
    
    r = l_size//s_size
    x = np.random.randint(img_size-s_size+1)
    y = np.random.randint(img_size-s_size+1)
    return input_big[:,:,x*r:x*r+l_size,y*r:y*r+l_size], input_sml[:,:,x:x+s_size,y:y+s_size]
        

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
    
    def forward(self, input_img, label):
        img256 = self.initial_conv(input_img)
        
        img128 = self.from256to128(img256)
        img64 = self.from128to64(img128)
        img32 = self.from64to32(img64)
        img16 = self.from32to16(img32)
        img8 = self.from16to8(img16)
                
        logits = self.final_conv(img8)
        
        if label == 'real':
            cropimg512, cropimg8 = crop(input_img, img16, 512, 8)
            decoded_img1 = self.decoder1(cropimg8)
            decoded_img2 = self.decoder2(img8)
            return logits, cropimg512, decoded_img1, decoded_img2

        return logits

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_image = torch.rand(1,3,1024,1024).to(device)
    D = Discriminator().to(device)
    output = D.forward(input_image, label='false')
    print(output)
    