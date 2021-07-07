import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.modules.activation import Softmax 

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()

        self.feature = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.remap = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.reconstruction = nn.Conv2d(32, 1, kernel_size=5, padding=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                nn.init.constant_(m.bias, val=0)

    def forward(self, input):
        x = self.feature(input)
        x = self.remap(x)
        x = self.reconstruction(x)
        return x



class Sample(nn.Module):
    def __init__(self):
        super(Sample, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(10, 15),
            nn.ReLU(),
            nn.Linear(15, 100),
            nn.Softmax()
        )

        self.w = nn.Conv2d(1, 64, 3, 1, 1)
        self.outt = nn.Linear(100, 254)
        for m in self.modules():
            print('Module ', m)
            if isinstance(m, nn.Linear):
                print('Isinstance', m)

    def forward(self, x):
        return self.net(self.w(x))


model = Sample()