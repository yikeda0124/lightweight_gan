import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt

from tqdm import tqdm

import argparse

from discriminator import Discriminator
from generator import Generator
from DiffAugment_pytorch import DiffAugment

def train(args):
    
    data_path = args.path
    image_size = args.im_size
    iterations = args.iter
    batch_size = args.batch_size
    latent_dim = args.latent
    
    policy = 'color, translation'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    G = Generator(latent_dim).to(device)
    D = Discriminator().to(device)
    
    for i in tqdm(range(iterations)):
        
        noise = torch.Tensor(batch_size, latent_dim, 1, 1).normal_(0, 1).to(device)
        gen_imgs = G(noise)
        
        fake_imgs = [DiffAugment(gi, policy=policy) for gi in gen_imgs] 
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='lightweight gan')
    
    parser.add_argument('path', type=str, help='path of dataset')
    parser.add_argument('--iter', type=int, default=50000, help='number of iterations')
    parser.add_argument('--im_size', type=int, default=1024, help='image resolution')
    parser.add_argument('--batch_size', type=int, default=8, help='mini batch size')
    parser.add_argument('--latent', type=int, default=256, help='number of latent dimentions of generator')
    
    args = parser.parse_args()
    
    train(args)