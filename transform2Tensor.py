import torch
from torch import nn 
from PIL import Image
import matplotlib.pyplot as plt 
import torchvision.transforms as transforms

loader = transforms.Compose([transforms.ToTensor()])

unloader = transforms.ToPILImage()

#input image path
#output tensor

device = torch.device("cuda")
def image_loader(image_name):
    image = Image.open(image_name).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

# b = torch.Tensor(1, 3)

# c = b.squeeze(0)

# print(c)

# def PIL_to_tensor(image):
#     image = loader(image).unsqueeze(0)
#     return image.to(device, torch.float)

# def tensor_to_PIL(tensor):
#     image = tensor.cpu().clone()
#     image = iamge.squeeze(0)
#     image = unloader(image)
#     return image

# def imshow(tensor, title=None):
#     image = tensor.cpu().clone()
#     image = image.squeeze(0)
#     image = unloader(image)
#     plt.imshow(image)
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001)

class ResnetBlock(nn.Module):
    def __init__(self, input_features, nb_features=6):
        super(ResnetBlock, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(input_features, nb_features, 3, 1, 1),
            nn.BatchNorm2d(nb_features),
            nn.LeakyReLU(),
            nn.Conv2d(nb_features, nb_features, 3, 1, 1),
            nn.BatchNorm2d(nb_features)
            )
        self.relu = nn.LeakyReLU()

        self.shortcut = nn.Sequential(
            nn.Conv2d(input_features, nb_features, 1, 1, bias=False),
            nn.BatchNorm2d(nb_features))

#     def weight_init(m):
# # 使用isinstance来判断m属于什么类型
#         if isinstance(m, nn.Conv2d):
#             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             m.weight.data.normal_(0, math.sqrt(2. / n))
#         elif isinstance(m, nn.BatchNorm2d):
#     # m中的weight，bias其实都是Variable，为了能学习参数以及后向传播
#             m.weight.data.fill_(1)
#             m.bias.data.zero_()
    # def _make_layer(self, input_features, nb_features, block_num, stride=1):
    #     shortcut = nn.Sequential(
    #         nn.Conv2d(input_features, nb_features, 1, stride, bias=False),
    #         nn.BatchNorm2d(nb_features)
    #         )

    #     layers = []
    #     layers.append(self.convs())

    def forward(self, x):
        convs = self.convs(x)
        x = self.shortcut(x)
        sum = convs + x 
        output = self.relu(sum)
        return output


img_tensor = image_loader(r'/home/user/neural-style/examples/1-content.jpg')
img_tensor_2 = image_loader(r'/home/user/neural-style/examples/1-output.jpg')

# net = ResnetBlock(3).to(device)
# img = net(img_tensor)
# # print(img_tensor.size())
# print(img)
# def imshow(tensor, title=None):
#     image = tensor.cpu().clone()
#     image = image.squeeze(0)
#     image = unloader(image)
#     plt.imshow(image)
#     if title is not None:
#         plt.title(title)
#     plt.pause(5)

# imshow(img)
loss = nn.L1Loss()
loss_data = loss(img_tensor, img_tensor_2)
print(loss_data.item())