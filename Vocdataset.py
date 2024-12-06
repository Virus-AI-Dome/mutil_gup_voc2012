# -*- coding: utf-8 -*-
# @Time    : 2024/11/28 下午5:19
# @Author  : ysj
# @FileName: Vocdataset.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/ydscc?type=blog


import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import xml.etree.ElementTree as ET
import cv2
class VocDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train'):
           self.root_dir = root_dir
           self.transform = transform
           self.split = split
           VOC_CLASSES = [
               '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor'
           ]

           # 创建标签名称到整数索引的映射
           self.class_to_idx = {cls: idx for idx, cls in enumerate(VOC_CLASSES)}
           #获取所以图像文件
           image_set_file = os.path.join(root_dir, 'ImageSets', 'Main',   f'{split}.txt')
           with open (image_set_file,'r') as f:
                self.images_ids = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.images_ids)

    def __getitem__(self, idx):
        image_id = self.images_ids[idx]
        # 加载图片
        img_path = os.path.join(self.root_dir, 'JPEGImages', f'{image_id}.jpg')
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        # 获取标注文件
        annotation_path = os.path.join(self.root_dir, 'Annotations', f'{image_id}.xml')
        boxes, labels = self.parse_annotations(annotation_path)



        if self.transform:
            image = self.transform(image)

        # 确保boxes和labels的尺寸正确，并返回
        return image, boxes, labels


    def parse_annotations(self, annotation_path):
         tree = ET.parse(annotation_path)
         root = tree.getroot()

         boxes = []
         labels = []

         for obj in root.iter('object'):
             # 获取框坐标
             bndbox = obj.find('bndbox')
             xmin = int(bndbox.find('xmin').text)
             ymin = int(bndbox.find('ymin').text)
             xmax = int(bndbox.find('xmax').text)
             ymax = int(bndbox.find('ymax').text)

             boxes.append([xmin, ymin, xmax, ymax])

             # 获取标签名称，并转换为对应的类别索引
             label = obj.find('name').text
             labels.append(self.class_to_idx[label])  # 使用类别索引

         return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)






