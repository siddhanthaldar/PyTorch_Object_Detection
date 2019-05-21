from __future__ import print_function

import os
import argparse
import itertools

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import numpy as np
from ssd import SSD300
from datagen import ListDataset
from multibox_loss import MultiBoxLoss

lr = 0.001
resume = False  # Resume from checkpoint
epoch = 200

use_cuda = torch.cuda.is_available()
best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last epoch

# Data
print('==> Preparing data..')
transform = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

# trainset = ListDataset(root='../data/train/images/', list_file='../data/train/annotations/', train=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=4)

# Model
net = SSD300()
# if resume:
#     print('==> Resuming from checkpoint..')
#     checkpoint = torch.load('../checkpoint/ckpt.pth')
#     net.load_state_dict(checkpoint['net'])
#     best_loss = checkpoint['loss']
#     start_epoch = checkpoint['epoch']
# else:
#     # Convert from pretrained VGG model.
#     try:
#         net.load_state_dict(torch.load('../model/ssd.pth'))
#         print('==> Pretrain model read successfully')
#     except:
#         print('==> Pretrain model read failed or not existed, training from init')
print("Loaded Model")



# criterion = MultiBoxLoss()

# if use_cuda:
#     net = torch.nn.DataParallel(net, device_ids=[0])
#     net.cuda()
#     cudnn.benchmark = True

# optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

# # Training
# def train(epoch):
#     print('\nEpoch: %d' % epoch)
#     net.train()
#     train_loss = 0
#     # for batch_idx, (images, loc_targets, conf_targets) in enumerate(trainloader):
#     #     if use_cuda:
#     #         images = images.cuda()
#     #         loc_targets = loc_targets.cuda()
#     #         conf_targets = conf_targets.cuda()

#     #     images = torch.tensor(images)
#     #     loc_targets = torch.tensor(loc_targets)
#     #     conf_targets = torch.tensor(conf_targets)

#     #     optimizer.zero_grad()
#     #     loc_preds, conf_preds = net(images)
#     #     loss = criterion(loc_preds, loc_targets, conf_preds, conf_targets)
#     #     loss.backward()
#     #     optimizer.step()

#     #     train_loss += loss.item()
#     #     if batch_idx%100 == 0:
#     #         os.makedirs('checkpoint', exist_ok=True)
#     #         torch.save({
#     #             'epoch': epoch,
#     #             'net': net.module.state_dict(), 
#     #             'loss': loss,
#     #         }, 'checkpoint/ckpt.pth')
#     #     print('epoch: {}, batch_idx: {},loss: {}, train_loss: {}'.format(epoch, batch_idx, loss.item(), train_loss/(batch_idx+1)))


# for epoch_num in range(1):#range(start_epoch, start_epoch+epoch):
#     train(epoch_num)