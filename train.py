# -*- coding: utf-8 -*-
# @Time    : 2024/11/28 下午5:47
# @Author  : ysj
# @FileName: train.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/ydscc?type=blog

import os
import torch
from torch.utils.data import  DataLoader
import torchvision.transforms as T
from torch.optim import  SGD
import time
from Vocdataset import VocDataset
import torchvision
from torchvision import models
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torch.optim import SGD
import time
from tqdm import tqdm
def get_model():

    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    num_classes =21

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    return model


def train(model, train_loader, optimizer, device):
    from torch.cuda.amp import autocast, GradScaler

    scaler = GradScaler()
    model.train()
    running_loss = 0.0
    for images, boxes, labels in tqdm(train_loader, desc="Training", ncols=100):

        images = [image.to(device) for image in images]
        boxes = [box.to(device) for box in boxes]
        labels = [label.to(device) for label in labels]

        # Zero gradients
        optimizer.zero_grad()

        with autocast():  # 自动混合精度
            loss_dict = model(images, targets=[{'boxes': box, 'labels': label} for box, label in zip(boxes, labels)])
            losses = sum(loss for loss in loss_dict.values())
        scaler.scale(losses).backward()  # 使用 scaler 处理反向传播
        scaler.step(optimizer)  # 更新优化器
        scaler.update()  # 更新 scaler
        running_loss += losses.item()

    return running_loss / len(train_loader)


def collate_fn(batch):
    return tuple(zip(*batch))
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据转换
    transform = T.Compose([T.ToTensor()])

    # 加载数据
    train_dataset = VocDataset(root_dir=r'/ysj_demo/ImgNet/VOCdevkit/VOC2012',
                               transform=transform, split='train')
    train_loader = DataLoader(train_dataset, batch_size=4,num_workers=48,shuffle=True,pin_memory=True, collate_fn=collate_fn)
    print("Train dataset size:", len(train_dataset))
    # 获取模型

    model = get_model()
    model.to(device)


    params = [p for p in model.parameters() if p.requires_grad]
    for param in model.backbone.parameters():
        param.requires_grad = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
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





