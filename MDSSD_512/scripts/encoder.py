'''Encode target locations and labels.'''
import torch

import math
import itertools

class DataEncoder:
    def __init__(self):
        '''Compute default box sizes with scale and aspect transform.'''
        scale = 512.
        steps = [s / scale for s in (32, 64, 128, 256, 512, 4, 8, 16)]
        sizes_ssd = [s / scale for s in (115.0, 230.4, 307.2, 384.0, 460.8, 537.6)]
        sizes_fusion = [s / scale for s in (35.84, 76.8, 153.6, 230.4)]
        aspect_ratios = ((2,3), (2,3), (2,), (2,), (2,), (2,), (2,),(2,))
        feature_map_sizes = (16,8,4,2,1,128,64,32)
        num_layers = len(feature_map_sizes)

        boxes = []
        for i in range(num_layers):
            fmsize = feature_map_sizes[i] # feature map size 
            for h,w in itertools.product(range(fmsize), repeat=2):
                cx = (w + 0.5)*steps[i]
                cy = (h + 0.5)*steps[i]

                if i<5:
                    s = sizes_ssd[i]
                    boxes.append((cx, cy, s, s))

                    s = math.sqrt(sizes_ssd[i] * sizes_ssd[i+1])
                    boxes.append((cx, cy, s, s))

                    s = sizes_ssd[i]
                    for ar in aspect_ratios[i]:
                        boxes.append((cx, cy, s * math.sqrt(ar), s / math.sqrt(ar)))
                        boxes.append((cx, cy, s / math.sqrt(ar), s * math.sqrt(ar)))

                else:            
                    s = sizes_fusion[i-5]
                    boxes.append((cx, cy, s, s))

                    s = math.sqrt(sizes_fusion[i-5] * sizes_fusion[i-5+1])
                    boxes.append((cx, cy, s, s))

                    s = sizes_fusion[i-5]
                    for ar in aspect_ratios[i]:
                        boxes.append((cx, cy, s * math.sqrt(ar), s / math.sqrt(ar)))
                        boxes.append((cx, cy, s / math.sqrt(ar), s * math.sqrt(ar)))

        self.default_boxes = torch.Tensor(boxes)

    def iou(self, box1, box2):
        '''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].

        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].

        Return:
          (tensor) iou, sized [N,M].
        '''
        N = box1.size(0)
        M = box2.size(0)

        lt = torch.max(
            box1[:,:2].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,:2].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        rb = torch.min(
            box1[:,2:].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,2:].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        wh[wh<0] = 0  # clip at 0
        inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

        area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # [N,]
        area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou

    def encode(self, boxes, classes, threshold=0.5):
        '''Transform target bounding boxes and class labels to SSD boxes and classes.

        Match each object box to all the default boxes, pick the ones with the
        Jaccard-Index > 0.5:
            Jaccard(A,B) = AB / (A+B-AB)

        Args:
          boxes: (tensor) object bounding boxes (xmin,ymin,xmax,ymax) of a image, sized [#obj, 4].
          classes: (tensor) object class labels of a image, sized [#obj,].
          threshold: (float) Jaccard index threshold

        Returns:
          boxes: (tensor) bounding boxes, sized [#obj, 8732, 4].
          classes: (tensor) class labels, sized [8732,]
        '''
        default_boxes = self.default_boxes
        num_default_boxes = default_boxes.size(0)
        num_objs = boxes.size(0)

        iou = self.iou(  # [#obj,8732]
            boxes,
            torch.cat([default_boxes[:,:2] - default_boxes[:,2:]/2,
                       default_boxes[:,:2] + default_boxes[:,2:]/2], 1)
        )

        iou, max_idx = iou.max(0)  # [1,8732]
        max_idx.squeeze_(0)        # [8732,]
        iou.squeeze_(0)            # [8732,]

        boxes = boxes[max_idx]     # [8732,4]
        variances = [0.1, 0.2]
        cxcy = (boxes[:,:2] + boxes[:,2:])/2 - default_boxes[:,:2]  # [8732,2]
        cxcy /= variances[0] * default_boxes[:,2:]
        wh = (boxes[:,2:] - boxes[:,:2]) / default_boxes[:,2:]      # [8732,2]
        wh = torch.log(wh) / variances[1]
        loc = torch.cat([cxcy, wh], 1)  # [8732,4]

        conf = 1 + classes[max_idx]   # [8732,], background class = 0
        conf[iou<threshold] = 0       # background
        return loc, conf

    def nms(self, bboxes, scores, threshold=0.3, mode='union'):
        '''Non maximum suppression.
        Args:
          bboxes: (tensor) bounding boxes, sized [N,4].
          scores: (tensor) bbox scores, sized [N,].
          threshold: (float) overlap threshold.
          mode: (str) 'union' or 'min'.

        Returns:
          keep: (tensor) selected indices.

        Ref:
          https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
        '''
        x1 = bboxes[:,0]
        y1 = bboxes[:,1]
        x2 = bboxes[:,2]
        y2 = bboxes[:,3]

        areas = (x2-x1) * (y2-y1)
        _, order = scores.sort(0, descending=True)
        order = order.numpy()

        keep = []
        while order.size > 0: 
            try:
                i = order[0]
            except:
                i = order
            keep.append(i)

            if order.size == 1:
                break

            xx1 = x1[order[1:]].clamp(min=x1[i])
            yy1 = y1[order[1:]].clamp(min=y1[i])
            xx2 = x2[order[1:]].clamp(max=x2[i])
            yy2 = y2[order[1:]].clamp(max=y2[i])

            w = (xx2-xx1).clamp(min=0)
            h = (yy2-yy1).clamp(min=0)
            inter = w*h

            if mode == 'union':
                ovr = inter / (areas[i] + areas[order[1:]] - inter)
            elif mode == 'min':
                ovr = inter / areas[order[1:]].clamp(max=areas[i])
            else:
                raise TypeError('Unknown nms mode: %s.' % mode)

            ids = (ovr<=threshold).nonzero().squeeze()
            if ids.size == 0:
                break
            order = order[ids+1]
        return torch.LongTensor(keep)

    def decode(self, loc, conf):
        '''Transform predicted loc/conf back to real bbox locations and class labels.

        Args:
          loc: (tensor) predicted loc, sized [8732,4].
          conf: (tensor) predicted conf, sized [8732,21].

        Returns:
          boxes: (tensor) bbox locations, sized [#obj, 4].
          labels: (tensor) class labels, sized [#obj,1].
        '''
        variances = (0.1, 0.2)
        wh = torch.exp(loc[:,2:]*variances[1]) * self.default_boxes[:,2:]
        cxcy = loc[:,:2] * variances[0] * self.default_boxes[:,2:] + self.default_boxes[:,:2]
        box_preds = torch.cat([cxcy-wh/2, cxcy+wh/2], 1)  # [8732,4]

        boxes = []
        labels = []
        scores = []
        num_classes = conf.size(1)
        for i in range(num_classes-1):
            score = conf[:,i+1]  # class i corresponds to (i+1) column
            mask = score > 0.1
            
            if not mask.any():
                continue
           
            box = box_preds[mask.nonzero().squeeze()]
            score = score[mask]
            
            if len(score) == 1:
                continue
            keep = self.nms(box, score, threshold=0.3)
            boxes.append(box[keep])
            labels.append(torch.LongTensor(len(box[keep])).fill_(i))
            scores.append(score[keep])

        return boxes, labels, scores
