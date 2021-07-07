import os 
from PIL import Image 
from torch.utils.data import Dataset 
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, dir, scale_factor, center_crop):
        self.dir = dir 
        self.img_list = os.listdir(self.dir)
        self.input_transform = transforms.Compose([
            transforms.CenterCrop(center_crop),
            transforms.Resize(center_crop//scale_factor),
            transforms.Resize(center_crop, interpolation=Image.BICUBIC),
            transforms.ToTensor()   
        ])

        self.target_transform = transforms.Compose([
            transforms.CenterCrop(center_crop),
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = Image.open(self.dir + '/' + self.img_list[idx]).convert('YCbCr')
        img, _, _ = img.split()
        lr_img = self.input_transform(img)
        or_img = self.target_transform(img)

        return lr_img, or_img