# -*- coding: utf-8 -*-
# @Time    : 2024/11/29 下午1:49
# @Author  : ysj
# @FileName: train_test.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/ydscc?type=blog

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET
from torchvision import transforms

# VOC 类别映射
VOC_CLASSES = [
    '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# 创建类别名称到索引的映射
class_to_idx = {cls: idx for idx, cls in enumerate(VOC_CLASSES)}


class VOCDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='trainval'):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split

        # 获取训练集或测试集的图像列表
        image_set_file = os.path.join(self.root_dir, 'ImageSets', 'Main', f'{split}.txt')
        with open(image_set_file, 'r') as f:
            self.image_ids = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        # 加载图片
        img_path = os.path.join(self.root_dir, 'JPEGImages', f'{image_id}.jpg')
        image = Image.open(img_path).convert("RGB")

        # 获取标注文件
        annotation_path = os.path.join(self.root_dir, 'Annotations', f'{image_id}.xml')
        boxes, labels = self.parse_annotations(annotation_path)

        if self.transform:
            image = self.transform(image)

        return image, boxes, labels

    def parse_annotations(self, annotation_path):
        """
        解析 VOC XML 标注文件，返回边界框和对应的标签索引。
        """
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
            labels.append(class_to_idx[label])  # 使用类别索引

        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)


import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn


def get_model(num_classes):
    # 加载预训练的 Faster R-CNN 模型
    model = fasterrcnn_resnet50_fpn(pretrained=True)

    # 获取模型的分类头，修改它来适应 VOC 类别
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    return model


import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
import torchvision.transforms as T
import time
from tqdm import tqdm

def train(model, train_loader, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, boxes, labels in tqdm(train_loader, desc="Training", ncols=100):
        images = [image.to(device) for image in images]
        boxes = [box.to(device) for box in boxes]
        labels = [label.to(device) for label in labels]

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        loss_dict = model(images, targets=[{'boxes': box, 'labels': label} for box, label in zip(boxes, labels)])

        # Compute total loss
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass
        losses.backward()
        optimizer.step()
        running_loss += losses.item()

    return running_loss / len(train_loader)


def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据转换
    transform = T.Compose([T.ToTensor()])

    # 加载数据
    train_dataset = VOCDataset(root_dir=r'/ysj_demo/ImgNet/VOCdevkit/VOC2012', transform=transform, split='trainval')
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    print("Train dataset size:", len(train_dataset))
    # 获取模型
    num_classes = len(VOC_CLASSES)
    print(F"{num_classes}")# VOC 共有 20 类 + 背景类
    model = get_model(num_classes)
    model.to(device)

    # 优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    print("Start training...")
    # 训练
    num_epochs = 10
    for epoch in range(num_epochs):
        start_time = time.time()
        loss = train(model, train_loader, optimizer, device)
        end_time = time.time()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}, Time: {end_time - start_time:.2f}s')

    # 保存模型
    torch.save(model.state_dict(), 'faster_rcnn_voc2012.pth')


if __name__ == '__main__':
    main()
