import cv2
import pandas as pd
import shutil
import os
import numpy as np
import glob
import xml.etree.ElementTree as ET
from xml.dom import minidom
import random

IMG_DIR = '../../../../VisDrone2019/dataset/VisDrone2018-DET-train/images/'
ANNOT_DIR = '../../../../VisDrone2019/dataset/VisDrone2018-DET-train/annotations/'

for i in os.listdir(ANNOT_DIR):
	box = []
	labels = []
	with open(os.path.join(ANNOT_DIR,i)) as f:
		f = f.read().split("\n")
		f = f[:-1]
	num_objs = len(f)

	for j in range(num_objs):
		f[j] = f[j].split(",")
		xmin = float(f[j][0])
		ymin = float(f[j][1])
		w = float(f[j][2])
		h = float(f[j][3])

		box.append([xmin,ymin,w,h])
		labels.append(int(f[j][5]))

	img = cv2.imread(IMG_DIR+i[:-4]+".jpg")
	box_new = box.copy()
	img_new = img.copy()
	# cv2.imshow("Image", img)
	# cv2.waitKey(0)
	
	for j in box:	
		if j[2]*j[3]<500:
			crop = img[int(j[1]):int(j[1]+j[3]),int(j[0]):int(j[0]+j[2])]
			x = random.randrange(0, img.shape[1],1)
			y = random.randrange(0, img.shape[0],1)

			try:
				img_new[int(y):int(y+j[3]),int(x):int(x+j[2])] = crop
				box_new.append([x,y,j[2],j[3]])
			except:
				continue
	for j in box_new:
		img_new = cv2.rectangle(img_new,(int(j[0]),int(j[1])),(int(j[0]+j[2]),int(j[1]+j[3])),(255,0,0),1)

	cv2.imshow("Image", img_new)
	cv2.waitKey(0)
	break