import math
import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt


class UpSampleConv(nn.Module):
    
    def __init__(self, chan_in, chan_out):
        super().__init__()
        
        self.upsample_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(chan_in, chan_out*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(chan_out*2),
            nn.GLU(dim=1)
        )
    
    def forward(self, input_img):
        return self.upsample_conv(input_img)

class SkipLayerExcitation(nn.Module):
    
    def __init__(self, chan_in, chan_out):
        super().__init__()
        
        self.skip_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(chan_in, chan_out, kernel_size=4),
            nn.LeakyReLU(0.1),
            nn.Conv2d(chan_out, chan_out, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, small_img, large_img):
        return large_img * self.skip_layer(small_img)

class FinalConv(nn.Module):
    
    def __init__(self, chan_in):
        super().__init__()
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(chan_in, 3, 3, 1, 1),
            nn.Tanh()
        )
        
    def forward(self, input_img):
        return self.final_conv(input_img)

class Generator(nn.Module):
    
    def __init__(self, latent_dim=256):
        super().__init__()
                
        self.initial_conv = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 1024*2, 4),
            nn.BatchNorm2d(1024*2),
            nn.GLU(dim=1)
        )
        
        self.from4to8 = UpSampleConv(1024, 512)
        self.from8to16 = UpSampleConv(512, 256)
        self.from16to32 = UpSampleConv(256, 128)
        self.from32to64 = UpSampleConv(128, 128)
        self.from64to128 = UpSampleConv(128, 64)
        self.from128to256 = UpSampleConv(64, 32)
        self.from256to512 = UpSampleConv(32, 16)
        self.from512to1024 = UpSampleConv(16, 8)
        
        self.sle128 = SkipLayerExcitation(512, 64)
        self.sle256 = SkipLayerExcitation(256, 32)
        self.sle512 = SkipLayerExcitation(128, 16)
        
        self.toimg1024 = FinalConv(8)
        
    def forward(self, input_latent):
        img4 = self.initial_conv(input_latent)
        
        img8 = self.from4to8(img4)
        img16 = self.from8to16(img8)
        img32 = self.from16to32(img16)
        img64 = self.from32to64(img32)
        
        img128_before = self.from64to128(img64)
        img128_after = self.sle128(img8, img128_before)
        
        img256_before = self.from128to256(img128_after)
        img256_after = self.sle256(img16, img256_before)
        
        img512_before = self.from256to512(img256_after)
        img512_after = self.sle512(img32, img512_before)
        
        img1024 = self.from512to1024(img512_after)
        img = self.toimg1024(img1024)
        
        return img


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_image = torch.rand(1,256,1,1).to(device)
    G = Generator().to(device)
    output_image = G.forward(input_image)
    img = output_image[0].cpu().detach().numpy().copy()
    plt.imshow(img.transpose(1, 2, 0))
    plt.show()