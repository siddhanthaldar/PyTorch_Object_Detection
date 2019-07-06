'''
Load image/class/box from a annotation file.

'''
from __future__ import print_function

import os
import sys
import os.path

import random
import numpy as np

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from encoder import DataEncoder
import cv2

import pandas as pd
import shutil
import os
import numpy as np
import glob
import xml.etree.ElementTree as ET
from xml.dom import minidom

class ListDataset(data.Dataset):
    img_size = 300

    def __init__(self, root, list_file, train, transform):
        '''
        Args:
          root: (str) ditectory to images.
          list_file: (str) path to annotation files.
          train: (boolean) train or test.
          transform: ([transforms]) image transforms.
        '''
        self.root = root
        self.train = train
        self.transform = transform

        self.fnames = []
        self.boxes = []
        self.labels = []

        self.data_encoder = DataEncoder()
        self.num_samples = 0

        # VisDrone

        for i in os.listdir(list_file):
            self.num_samples += 1
            self.fnames.append(i)
            box = []
            labels = []
            with open(os.path.join(list_file,i)) as f:
                f = f.read().split("\n")
                f = f[:-1]
            num_objs = len(f)

            for j in range(num_objs):
                f[j] = f[j].split(",")
                xmin = float(f[j][0])
                ymin = float(f[j][1])
                w = float(f[j][2])
                h = float(f[j][3])

                box.append([xmin,ymin,xmin+h,ymin+h])
                labels.append(int(f[j][5]))
        
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(labels))
        

    def __getitem__(self, idx):
        '''Load a image, and encode its bbox locations and class labels.
        Args:
          idx: (int) image index.
        Returns:
          img: (tensor) image tensor.
          loc_target: (tensor) location targets, sized [8732,4].
          conf_target: (tensor) label targets, sized [8732,].
        '''
        # Load image and bbox locations.
        fname = self.fnames[idx]
        img = cv2.imread(os.path.join(self.root, fname[:-4]+".jpg"))
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx]

        # Data augmentation while training.
        if self.train:
            img, boxes = self.random_flip(img, boxes)
            img, boxes, labels = self.random_crop(img, boxes, labels)

        # Scale bbox locaitons to [0,1].
        w,h = img.shape[1], img.shape[0]
        boxes /= torch.Tensor([w,h,w,h]).expand_as(boxes)
        img = cv2.resize(img, (self.img_size,self.img_size))
        img = self.transform(img)

        # Encode loc & conf targets.
        
        loc_target, conf_target = self.data_encoder.encode(boxes, labels)
        return img, loc_target, conf_target

    def random_flip(self, img, boxes):
        '''Randomly flip the image and adjust the bbox locations.
        For bbox (xmin, ymin, xmax, ymax), the flipped bbox is:
        (w-xmax, ymin, w-xmin, ymax).
        Args:
          img: (ndarray.Image) image. f
          boxes: (tensor) bbox locations, sized [#obj, 4].
        Returns:
          img: (ndarray.Image) randomly flipped image.
          boxes: (tensor) randomly flipped bbox locations, sized [#obj, 4].
        '''
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
            w = img.shape[1]
            xmin = w - boxes[:,2]
            xmax = w - boxes[:,0]
            boxes[:,0] = xmin
            boxes[:,2] = xmax
        return img, boxes

    def random_crop(self, img, boxes, labels):
        '''Randomly crop the image and adjust the bbox locations.
        For more details, see 'Chapter2.2: Data augmentation' of the paper.
        Args:
          img: (ndarray.Image) image.
          boxes: (tensor) bbox locations, sized [#obj, 4].
          labels: (tensor) bbox labels, sized [#obj,].
        Returns:
          img: (ndarray.Image) cropped image.
          selected_boxes: (tensor) selected bbox locations.
          labels: (tensor) selected bbox labels.
        '''
        imw, imh = img.shape[1], img.shape[0]
        while True:
            min_iou = random.choice([None, 0.1, 0.3, 0.5, 0.7, 0.9])# random choice the one 
            if min_iou is None:
                return img, boxes, labels

            for _ in range(100):
                w = random.randrange(int(0.1*imw), imw)
                h = random.randrange(int(0.1*imh), imh)

                if h > 2*w or w > 2*h or h < 1 or w < 1:
                    continue

                x = random.randrange(imw - w)
                y = random.randrange(imh - h)
                roi = torch.Tensor([[x, y, x+w, y+h]])
                
                center = (boxes[:,:2] + boxes[:,2:]) / 2  # [N,2]
                roi2 = roi.expand(len(center), 4)  # [N,4]
    
                mask = (center > roi2[:,:2]) & (center < roi2[:,2:])  # [N,2]
                mask = mask[:,0] & mask[:,1]  #[N,]

                if not mask.any():
                    continue
              
                selected_boxes = boxes.index_select(0, mask.nonzero().squeeze(1))
                
                iou = self.data_encoder.iou(selected_boxes, roi)
                if iou.min() < min_iou:
                    continue
                img = img[y:y+h, x:x+w, :]
                
                selected_boxes[:,0].add_(-x).clamp_(min=0, max=w)
                selected_boxes[:,1].add_(-y).clamp_(min=0, max=h)
                selected_boxes[:,2].add_(-x).clamp_(min=0, max=w)
                selected_boxes[:,3].add_(-y).clamp_(min=0, max=h)

                return img, selected_boxes, labels[mask]

    def __len__(self):
        return self.num_samples