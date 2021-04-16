import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm

import argparse

from discriminator import Discriminator
from generator import Generator
from dataloader import CustomImageDataset
from DiffAugment_pytorch import DiffAugment

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            m.weight.data.normal_(0.0, 0.02)
        except:
            pass
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Train():
    
    def __init__(self, args):
        root = args.root
        im_size = args.im_size
        batch_size = args.batch_size
        
        self.iterations = args.iter
        self.latent_dim = args.latent
    
        self.policy = 'color,translation'
    
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.G = Generator(self.latent_dim).to(self.device)
        self.D = Discriminator().to(self.device)
        
        self.G.apply(weights_init)
        self.D.apply(weights_init)
        
        self.G_optim = optim.Adam(self.G.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.D_optim = optim.Adam(self.D.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        trans_list = [
            transforms.Resize((im_size, im_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
        trans = transforms.Compose(trans_list)
        
        dataset = CustomImageDataset(root=root, transform=trans)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        
        self.mse = nn.MSELoss()
        
    def train(self):
        for i in tqdm(range(self.iterations)):
            real_imgs = next(iter(self.dataloader))
            real_imgs = real_imgs.to(self.device, non_blocking=True)
            real_imgs = DiffAugment(real_imgs, policy=self.policy)
            
            cur_batch_size = real_imgs.shape[0]
            noise = torch.Tensor(cur_batch_size, self.latent_dim, 1, 1).normal_(0, 1).to(self.device, non_blocking=True)
            gen_imgs = self.G(noise)
            
            fake_imgs = DiffAugment(gen_imgs, policy=self.policy)
            
            self.D.zero_grad()
            self.train_discriminator(real_imgs, label='real')
            self.train_discriminator(fake_imgs, label='fake')
            self.D_optim.step()
            
            self.G.zero_grad()
            pred = self.D(fake_imgs, label='fake')
            loss3 = -pred.mean()
            loss3.backward()
            self.G_optim.step()
        
            
            if i % 5000 == 0:
                model_path = 'model' + str(i) + '.pth' 
                torch.save(self.G.state_dict(), model_path)
        
        
        noise = torch.Tensor(cur_batch_size, self.latent_dim, 1, 1).normal_(0, 1).to(self.device, non_blocking=True)
        gen_imgs = self.G(noise)
        img = gen_imgs[0].cpu().detach().numpy().copy()
        plt.imshow(img.transpose(1, 2, 0))
        plt.show()
    
    def train_discriminator(self, image, label):
        if label == 'real':
            logits, cropimg512, decoded_img1, decoded_img2 = self.D(image, label)
            loss = F.relu(torch.rand_like(logits)*0.2 + 0.8 - logits).mean() + \
                self.mse(decoded_img1, F.interpolate(image, decoded_img1.shape[2])).sum() + \
                self.mse(decoded_img2, F.interpolate(cropimg512, decoded_img2.shape[2])).sum()
            loss.backward(retain_graph=True)

        else:
            logits = self.D(image, label)
            loss = F.relu(torch.rand_like(logits)*0.2 + 0.8 + logits).mean()
            loss.backward(retain_graph=True)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='lightweight gan')
    
    parser.add_argument('root', type=str, help='root of dataset')
    parser.add_argument('--iter', type=int, default=1000, help='number of iterations')
    parser.add_argument('--im_size', type=int, default=1024, help='image resolution')
    parser.add_argument('--batch_size', type=int, default=8, help='mini batch size')
    parser.add_argument('--latent', type=int, default=256, help='number of latent dimentions of generator')
    
    args = parser.parse_args()
    
    t = Train(args)
    t.train()