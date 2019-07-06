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
from mdssd import MDSSD300
from datagen import ListDataset
from multibox_loss import MultiBoxLoss

lr = 0.001
resume = False  # Resume from checkpoint
epoch = 200
batch_size = 8

use_cuda = torch.cuda.is_available()
best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last epoch

# Data
print('==> Preparing data..')
transform = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize(mean=(0.356, 0.368, 0.362), std=(0.242, 0.235, 0.236))])

trainset = ListDataset(root='../../../../VisDrone2019/dataset/VisDrone2018-DET-train/images/', list_file='../../../../VisDrone2019/dataset/VisDrone2018-DET-train/annotations/', train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=4)

valset = ListDataset(root='../../../../VisDrone2019/dataset/VisDrone2019-DET-val/images/', list_file='../../../../VisDrone2019/dataset/VisDrone2019-DET-val/annotations/', train=True, transform=transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=8, shuffle=True, num_workers=4)

# Model
net = MDSSD300()
if resume:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('./checkpoint/ckpt.pth')

    keys = []
    for k,v in checkpoint['net'].items():
        if "module" in k:
            keys.append(k)
    for i in keys:
        checkpoint['net'][i[7:]] = checkpoint['net'][i]
        del checkpoint['net'][i]

    net.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']
else:
    # Convert from pretrained VGG model.
    try:
        net.load_state_dict(torch.load('../model/ssd.pth'))
        print('==> Pretrain model read successfully')
    except:
        print('==> Pretrain model read failed or not existed, training from init')

criterion = MultiBoxLoss()

if use_cuda:
    net = torch.nn.DataParallel(net, device_ids=[0])
    net.cuda()
    cudnn.benchmark = True

optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.25)

# Training
def train(epoch,prev_val_loss, last_saved):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    for batch_idx, (images, loc_targets, conf_targets) in enumerate(trainloader):
        if use_cuda:
            images = images.cuda()
            loc_targets = loc_targets.cuda()
            conf_targets = conf_targets.cuda()

        images = torch.tensor(images)
        loc_targets = torch.tensor(loc_targets)
        conf_targets = torch.tensor(conf_targets)

        optimizer.zero_grad()
        loc_preds, conf_preds = net(images)
        loss = criterion(loc_preds, loc_targets, conf_preds, conf_targets)
        loss.backward()
        optimizer.step()
        # scheduler.step()

        train_loss += loss.item()

        if batch_idx%100 == 0:
            val_loss_tot = 0
            for batch_idx_val, (images, loc_targets, conf_targets) in enumerate(valloader):
                if use_cuda:
                    images = images.cuda()
                    loc_targets = loc_targets.cuda()
                    conf_targets = conf_targets.cuda()

                images = torch.tensor(images)
                loc_targets = torch.tensor(loc_targets)
                conf_targets = torch.tensor(conf_targets)

                loc_preds, conf_preds = net(images)
                val_loss = criterion(loc_preds, loc_targets, conf_preds, conf_targets)
                val_loss_tot += val_loss.item()

            val_loss_tot /= (batch_idx_val+1)
            if val_loss_tot < prev_val_loss:
                os.makedirs('checkpoint', exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'net': net.state_dict(), 
                    'loss': loss,
                }, 'checkpoint/ckpt.pth')
                print("Saved.")
                prev_val_loss = val_loss_tot
                last_saved = [epoch, batch_idx]

        print('epoch: {}, batch_idx: {},loss: {}, train_loss: {}, best_val_loss: {}, last_saved: {}'.format(epoch, batch_idx, loss.item(), train_loss/(batch_idx+1), prev_val_loss, last_saved))

    return prev_val_loss, last_saved


prev_val_loss = 999
last_saved = [start_epoch,0]
for epoch_num in range(start_epoch, start_epoch+epoch):
    prev_val_loss, last_saved = train(epoch_num, prev_val_loss, last_saved)
