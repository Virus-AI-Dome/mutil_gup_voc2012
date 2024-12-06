import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
import cv2
from torchvision import models
# 类别名称到索引的映射
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
class VOCDetectionDataset(Dataset):
    def __init__(self, root, year="2012", image_set="train", transforms=None):
        """
        Args:
            root (string): 根目录，指向 VOCdevkit 文件夹。
            year (string): 选择 "2012" 或 "2007"。
            image_set (string): "train", "val", "trainval" 或 "test"。
            transforms (callable, optional): 可选的预处理函数（例如 ToTensor()）。
        """
        VOC_CLASSES = [
    '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

        self.class_to_idx = {cls: idx for idx, cls in enumerate(VOC_CLASSES)}
        self.root = root
        self.year = year
        self.image_set = image_set
        self.transforms = transforms
        
        # 获取指定图片集的文件列表
        self.image_set_path = os.path.join(self.root, f"VOC{self.year}", "ImageSets", "Main", f"{self.image_set}.txt")
        with open(self.image_set_path, 'r') as f:
            self.image_ids = [x.strip() for x in f.readlines()]
        
        # 获取图像路径和标注路径
        self.images_path = os.path.join(self.root, f"VOC{self.year}", "JPEGImages")
        self.annotations_path = os.path.join(self.root, f"VOC{self.year}", "Annotations")
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        # 获取当前图像的 ID
        image_id = self.image_ids[idx]
        
        # 加载图像
        image = cv2.imread(os.path.join(self.images_path, f"{image_id}.jpg"))
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        
        # 加载标注文件
      
        annotation = self.parse_annotations(image_id)

        # 应用预处理
        image = F.to_tensor(image)
        
        return image, annotation
    
    def parse_annotations(self, image_id):
         annotation_path = os.path.join(self.annotations_path, f"{image_id}.xml")
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
             target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64)}
         return target



# 加载 VOC 数据集
def get_voc_dataset(root, year="2012", split="train"):
    transform = transforms.Compose([transforms.ToTensor(),])
    dataset = VOCDetectionDataset(
    root="/ysj_demo/ImgNet/VOCdevkit", 
    year="2012", 
    image_set="train", 
    transforms=transform
)
    return dataset

def collate_fn(batch):
    return tuple(zip(*batch))
# 数据加载器
def get_dataloader(dataset, batch_size, num_workers, distributed):
    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        sampler=sampler
    )
    return dataloader


from torchvision.models.detection import fasterrcnn_resnet50_fpn

def get_model(num_classes):
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    num_classes =21

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    return model


import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import os

def train(rank, world_size, args):
    device = torch.device(f'cuda:{rank}')
    # 初始化分布式环境
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # 加载数据集和数据加载器
    dataset = get_voc_dataset('/ysj_demo/ImgNet')
    dataloader = get_dataloader(dataset, args.batch_size, args.num_workers, distributed=True)

    # 模型定义
    model = get_model(num_classes=21)  # VOC2012 有 20 个类别 + 背景
    model.to(rank)
    model = DDP(model, device_ids=[rank])

    # 优化器和学习率调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    scaler = GradScaler()

    # 训练循环
    
    for epoch in range(args.epochs):
        dataloader.sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        accumulation_steps = 64
        pbar = tqdm(dataloader, desc=f"Rank {rank} Training Epoch {epoch}", position=rank, ncols=100)
        i= 0 
        for  images, targets in pbar:
            i+=1
            images = [image.to(rank) for image in images]
            targets = [{k: v.to(rank) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            with autocast():
                losses = model(images, targets)
                loss = sum(loss for loss in losses.values())
            
            scaler.scale(loss).backward()
          
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / len(dataloader))
            torch.cuda.empty_cache()
    
    # 清理分布式环境
    dist.destroy_process_group()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='/ysj_demo/ImgNet', help="Path to VOC2012 dataset root.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU.")
    parser.add_argument("--num_workers", type=int, default=48, help="Number of data loader workers.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "127.0.0.0"
    os.environ["MASTER_PORT"] = "29510"

    torch.multiprocessing.spawn(
        train,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )



