import torch
from torch import nn 
from PIL import Image
import matplotlib.pyplot as plt 
import torchvision.transforms as transforms

device = torch.device("cuda")
loader = transforms.Compose([transforms.ToTensor()])
unloader = transforms.ToPILImage()

class Discriminator(nn.Module):
    def __init__(self, input_features):
        super(Discriminator, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(input_features, 96, 3, 2, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 64, 3, 2, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),

            nn.AvgPool2d(3, 2, 1),

            nn.Conv2d(64, 32, 3, 1, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 1, 1, 0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),

            nn.AvgPool2d(3, 2, 1),

            nn.Conv2d(32, 2, 1, 1, 0),
            nn.LeakyReLU()
        )

    def forward(self, x):
        convs = self.convs(x)
        output = convs.view(convs.size(0), -1, 2)
        return convs, output

def image_loader(image_name):
    image = Image.open(image_name).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

img_tensor = image_loader(r'/home/user/1-style.jpg')
print(img_tensor.size())
net = Discriminator(3).to(device)
img_pre, img = net(img_tensor)
print(img_pre.size())
print(img_pre)
print(img.size())
print(img)