import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms

import sys
from mdssd import MDSSD300
from encoder import DataEncoder
import cv2

VOC_LABELS = (
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


# Load model
net = MDSSD300()
#net.load_state_dict(torch.load('model/net.pth'))
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

if len(sys.argv) == 2:
    img_path = sys.argv[1]
else:
    img_path = './images/img1.jpg'
# Load test image
#img = Image.open(img_path)
#img1 = img.resize((300,300))
img = cv2.imread(img_path)
# cv2.imshow("Orig",img)
# cv2.waitKey(0)
img1 = cv2.resize(img, (300, 300))
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
img1 = transform(img1)

# Forward
with torch.no_grad():
    x = torch.tensor(img1)
    loc_preds, conf = net(x.unsqueeze(0))
# Decode
data_encoder = DataEncoder()
boxes, labels, scores = data_encoder.decode(loc_preds.data.squeeze(0), F.softmax(conf.squeeze(0), dim=1).data)
for box, label, score in zip(boxes, labels, scores):
    box[::2] *= img.shape[1]
    box[1::2] *= img.shape[0]
    for b, s in zip(box, score):
        if s > 0.25:#0.7:
            print('label:',VOC_LABELS[int(label[0])], 'score:', score)
            b = list(b)
            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (255, 255, 255), 2)
            title = '{}: {}'.format(VOC_LABELS[int(label[0])], round(float(score[0]), 2))
            cv2.putText(img, title, (b[0], b[1]), cv2.FONT_ITALIC, 0.6, (0, 255, 0), 2)
            cv2.imshow('img', img)
cv2.waitKey(0)