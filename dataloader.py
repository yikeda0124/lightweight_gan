import os
from torch.utils.data import Dataset
from torchvision.io import read_image
from PIL import Image

class CustomImageDataset(Dataset):
    
    def __init__(self, root, transform=None):
        super().__init__()
        self.root = root
        self.frame = self._parse_frame()
        self.transform = transform
        
    def _parse_frame(self):
        frame = []
        img_names = os.listdir(self.root)
        img_names.sort()
        for i in range(len(img_names)):
            image_path = os.path.join(self.root, img_names[i])
            if image_path[-4:] == '.jpg' or image_path[-4:] == '.png' or image_path[-5:] == '.jpeg':
                frame.append(image_path)
        return frame
        
    def __len__(self):
        return len(self.frame)
    
    def __getitem__(self, idx):
        path = self.frame[idx]
        img = Image.open(path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img