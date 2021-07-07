import torch 
from torchvision import transforms
from PIL import Image
from models import SRCNN
import numpy as np 

checkpoint = torch.load('model.pth')
input_file = 't1.png'
model = SRCNN()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
zoom_factor = 2

img = Image.open(input_file).convert('YCbCr')
img = img.resize((int(img.width * zoom_factor), int(img.height * zoom_factor)), Image.BICUBIC)
y, cb, cr = img.split()
img_to_tensor = transforms.ToTensor()

print(y.size)

img = img_to_tensor(y).view(1, -1, y.size[1], y.size[0])


out_srcnn = model(img)
out_img = out_srcnn[0].detach().numpy()
out_img *= 255.0
out_img = out_img.clip(0, 255)
out_img = Image.fromarray(np.uint8(out_img[0]), mode='L')

out_img_final = Image.merge('YCbCr', [out_img, cb, cr]).convert('RGB')

out_img_final.save('srcnn.png')