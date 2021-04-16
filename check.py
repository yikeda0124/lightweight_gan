import torch
from generator import Generator
import matplotlib.pyplot as plt


class Check:
    
    def __init__(self):
        
        self.latent_dim = 256
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.G = Generator(self.latent_dim).to(self.device)
        
        model_path = 'model0.pth'
        self.G.load_state_dict(torch.load(model_path))
        
    def check(self):
        noise = torch.Tensor(1, self.latent_dim, 1, 1).normal_(0, 1).to(self.device, non_blocking=True)
        gen_imgs = self.G(noise)
        img = gen_imgs[0].cpu().detach().numpy().copy()
        plt.imshow(img.transpose(1, 2, 0))
        plt.show()
    
        

if __name__ == '__main__':
    
    c = Check()
    c.check()