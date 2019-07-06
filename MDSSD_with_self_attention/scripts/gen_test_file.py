import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms

import sys
from mdssd import MDSSD300
from encoder import DataEncoder
import cv2

import pandas as pd
import shutil
import os
import numpy as np
import glob
import xml.etree.ElementTree as ET
from xml.dom import minidom

TEST_DIR = '/home/siddhant/deeplearning/Dataset/VisDrone2019/VisDrone2019-DET-val/images/'
TEST_ANNOT = '/home/siddhant/deeplearning/Dataset/VisDrone2019/VisDrone2019-DET-val/annotations/'

LABELS = (
	'ignored regions',
	'pedestrian',
	'people',
	'bicycle',
	'car',
	'van',
	'truck',
	'tricycle',
	'awning-tricycle',
	'bus',
	'motor',
	'other'
)

def GT(annotation_file):
	# Load model
	net = MDSSD300()
	checkpoint = torch.load('./checkpoint/ckpt.pth')

	keys = []
	for k,v in checkpoint['net'].items():
		if "module" in k:
			keys.append(k)
	for i in keys:
		checkpoint['net'][i[7:]] = checkpoint['net'][i]
		del checkpoint['net'][i]

	net.load_state_dict(checkpoint['net'])
	net.eval()

	count = 0
	for i in os.listdir(annotation_file):
		count += 1
		print(count)
		with open(os.path.join(annotation_file,i)) as f:
			f = f.read().split("\n")
			f = f[:-1]
		num_objs = len(f)

		file = open(os.path.join("../test/gt/",i[:-4]+".txt"), "w")

		for j in range(num_objs):
			f[j] = f[j].split(",")
			label = int(f[j][5])
			if label == 0:
				continue
			xmin = float(f[j][0])
			ymin = float(f[j][1])
			w = float(f[j][2])
			h = float(f[j][3])
			file.write(str(LABELS[label])+"	"+str(int(xmin))+"	"+str(int(ymin))+"	"+str(int(xmin+w))+"	"+str(int(ymin+h))+"\n")
		file.close()

def detect(image_dir):
	# Load model
	net = MDSSD300()
	checkpoint = torch.load('./checkpoint/ckpt.pth')

	keys = []
	for k,v in checkpoint['net'].items():
		if "module" in k:
			keys.append(k)
	for i in keys:
		checkpoint['net'][i[7:]] = checkpoint['net'][i]
		del checkpoint['net'][i]

	net.load_state_dict(checkpoint['net'])
	net.eval()

	count = 0
	for i in os.listdir(image_dir):
		count += 1
		print(count)
		file = open("../test/detect/"+i[:-4]+".txt","w")
		img = cv2.imread(os.path.join(image_dir,i))

		img1 = cv2.resize(img, (300, 300))
		transform = transforms.Compose([transforms.ToTensor(),
										transforms.Normalize(mean=(0.356, 0.368, 0.362), std=(0.242, 0.235, 0.236))])
		img1 = transform(img1)

		# Forward
		with torch.no_grad():
			x = torch.tensor(img1)
			loc_preds, conf = net(x.unsqueeze(0))
		# Decode
		data_encoder = DataEncoder()
		boxes, labels, scores = data_encoder.decode(loc_preds.data.squeeze(0), F.softmax(conf.squeeze(0), dim=1).data)

		for box, label, score in zip(boxes, labels, scores):
			for b, l, s in zip(box, label, score):
				# print(b,l,s)
				if l.item() == 0:
					continue
				b[::2] *= img.shape[1]
				b[1::2] *= img.shape[0]

				xmin = str(int(b[0].item()))
				ymin = str(int(b[1].item()))
				xmax = str(int(b[2].item()))
				ymax = str(int(b[3].item()))
				confidence = str(s.item())
				label = str(LABELS[int(l.item())])
				file.write(label+"	"+confidence+"	"+xmin+"	"+ymin+"	"+xmax+"	"+ymax+"\n")

		file.close()


if __name__ == "__main__":
	GT(TEST_ANNOT)
	detect(TEST_DIR)
