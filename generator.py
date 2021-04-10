import torch
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Generator(nn.Module):
    
    def __init__(self, latent_dim=256):
        super().__init__()
        
        self.initial_conv = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, latent_dim*2, 4),
            nn.BatchNorm2d(latent_dim*2),
            nn.GLU(dim=1)
        )

if __name__ == '__main__':
    print(device)
    model = Generator().to(device)
    print(model)