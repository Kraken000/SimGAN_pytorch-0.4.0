import torch
import torch.utils.data as Data 
import torchvision
from torch import nn 
from torchvision import transforms
#from lib.image_history_buffer import ImageHistoryBuffer 

class Main(object):
    def __init__(self):
        #network
        self.R = None
        self.D = None
        self.opt_R = None
        self.opt_D = None
        self.self_regularization_loss = None
        self.local_adversarial_loss = None
        self.delta = None

        #data
        self.syn_train_loader = None
        self.real_loader = None

    device = torch.device("cuda")

    def build_network(self):
        print('=' * 55)
        print('Building network...')
        self.R = Refiner(4, cfg.img_channels, nb_features=64)
        self.D = Discriminator(input_features=cfg.img_channels)

        if cfg.cuda_use:
            self.R.cuda(cfg.cuda_num)
            self.D.cuda(cfg.cuda_num)

        self.opt_R = torch.optim.Adam(self.R.parameters(), lr=cfg.r_lr)
        self.opt_D = torch.optim.SGD(self.D.parameters(), lr=cfg.d_lr)
        self.self_regularization_loss = nn.L1Loss(size_average=False)
        self.local_adversarial_loss = nn.CrossEntropyLoss(size_average=True)
        self.delta = cfg.delta

    def load_data(self):
        print('=' * 50)
        print('Loading data...')
        transforms = transforms.Compose([
            transforms.ImageOps.grayscale, 
            transforms.Scale((cfg.img_width, cfg.img_height)),
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        syn_train_folder = torchvision.datasets.ImageFolder(root=cfg.syn_path, transform=transforms)
        self.syn_train_loader = Data.DataLoader(syn_train_folder, batch_size=cfg.batch_size, shuffle=True,
                                                pin_memory=True)
        print('syn_train_batch %d' % len(self.syn_train_loader))  

        real_folder = torchvision.datasets.ImageFolder(root=cfg.real_path, transform=transforms)
        self.real_loader = Data.DataLoader(real_folder, batch_size=cfg.batch_size, shuffle=True, pin_memory=True)

        print('real_batch %d' % len(self.real_loader))

    def pre_train_r(self):
        print('=' * 50)
        if cfg.ref_pre_path:
            print('Loading R_pre from %s' % cfg.ref_pre_path)
            self.R.load_state_dict(torch.load(cfg.ref_pre_path))
            return

        print('pre-training the refiner network %d times...' % cfg.r_pretrain)

        for index in range(cfg.r_pretrain):
            syn_image_batch, _ = self.syn_train_loader.__iter__().next()
            syn_image_batch = syn_image_batch.to(device)

            self.R.train()
            ref_image_batch = self.R(syn_image_batch)

            r_loss = self.self_regularization_loss(ref_image_batch, syn_image_batch)
            r_loss = torch.mul(r_loss, self.delta)

            self.opt_R.zero_grad()
            r_loss.backward()
            self.opt_R.step()

        # log every `log_interval` steps
        if (index % cfg.r_pre_per == 0) or (index == cfg.r_pretrain - 1):
            print('[%d/%d] (R)reg_loss: %.4f' % (index, cfg.r_pretrain, r_loss.data[0]))

            with torch.no_grad():
                syn_image_batch, _ = self.syn_train_loader.__iter__().next()
                real_image_batch = self.real_loader.__iter__().next()

                self.R.eval()
                ref_image_batch = self.R(syn_image_batch)

                figure_path = os.path.join(cfg.train_res_path, 'refined_image_batch_pre_train_%d.png' % index)
                generate_img_batch(syn_image_batch.data.cpu(), ref_image_batch.data.cpu(),
                                    real_image_batch.data, figure_path)

            


