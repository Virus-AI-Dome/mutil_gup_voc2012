# import os
# import torch
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torchvision import models
# from torch.utils.data import DataLoader
# from torch.utils.data.distributed import DistributedSampler
# from torchvision.datasets import VOCDetection
# from torchvision.transforms import functional as F
# from torchvision.transforms import ToTensor
# import torchvision.transforms as T
# import torch.optim as optim
# from torch import nn
# import time

# def setup(rank, world_size):


#     dist.init_process_group("nccl", rank=rank, world_size=world_size)
#     torch.cuda.set_device(rank)

# def cleanup():
#     dist.destroy_process_group()

# def collate_fn(batch):
#     return tuple(zip(*batch))

# class VOCTransform:
#     def __call__(self, image, target):
#         image = F.to_tensor(image)
#         objects = target['annotation']['object']
#         boxes = []
#         labels = []
#         for obj in objects:
#             bbox = obj['bndbox']
#             boxes.append([
#                 float(bbox['xmin']),
#                 float(bbox['ymin']),
#                 float(bbox['xmax']),
#                 float(bbox['ymax'])
#             ])
#             labels.append(1)  # 你可以根据需要映射为类别索引
#         target = {
#             'boxes': torch.tensor(boxes, dtype=torch.float32),
#             'labels': torch.tensor(labels, dtype=torch.int64)
#         }
#         return image, target

# def train(world_size):
#     # setup(rank, world_size)
#     rank = torch.cuda.device_count()
#     if nranks > 1 :
#         # 初始化NCCL环境
#         dist.init_process_group(backend='nccl')
#         local_rank = int(os.environ["LOCAL_RANK"])

#     # 加载 Faster R-CNN 模型
#     model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(local_rank)
#     model = DDP(model, device_ids=[local_rank])

#     # 数据加载
#     dataset = VOCDetection(
#         root="/path/to/VOCdevkit",
#         year="2012",
#         image_set="train",
#         download=True,
#         transforms=VOCTransform()
#     )

#     torch.nn.parallel.DistributedDataParallel(model, gradient_as_bucket_view=True)
#     dataloader = DataLoader(dataset, batch_size=4, num_workers=4, pin_memory=True, sampler=sampler, collate_fn=collate_fn)

#     # 优化器和学习率调度器
#     optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
#     lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

#     model.train()

#     for epoch in range(10):  # 假设训练 10 个 epoch
#         sampler.set_epoch(epoch)
#         for i, (images, targets) in enumerate(dataloader):
#             s= time.time()
#             images = [image.to(local_rank) for image in images]
#             targets = [{k: v.to(local_rank) for k, v in t.items()} for t in targets]

#             optimizer.zero_grad()
#             loss_dict = model(images, targets)
#             losses = sum(loss for loss in loss_dict.values())
#             losses.backward()
#             optimizer.step()

#             if i % 100 == 0:
#                 print(f"Rank {local_rank}, Epoch [{epoch+1}/10], Step [{i}/{len(dataloader)}], Loss: {losses.item()}")

#         lr_scheduler.step()

#     cleanup()

# if __name__ == "__main__":
#     # world_size = torch.cuda.device_count()  # 获取 GPU 数量
#     # torch.multiprocessing.spawn(train, args=(4,), nprocs=4, join=True)
#     train(4)
import os
from torchvision import transforms as T 
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.datasets import VOCDetection
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import time
from torch.distributed import barrier
# 1. 初始化分布式环境
def setup_distributed():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    return local_rank

# 2. 定义模型
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    # 替换分类头
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# 3. 数据加载器
class VOCTransform:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        objects = target['annotation']['object']
        boxes = []
        labels = []
        for obj in objects:
            bbox = obj['bndbox']
            boxes.append([
                float(bbox['xmin']),
                float(bbox['ymin']),
                float(bbox['xmax']),
                float(bbox['ymax'])
            ])
            labels.append(1)  # 你可以根据需要映射为类别索引
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64)
        }
        return image, target

def collate_fn(batch):
    return tuple(zip(*batch))

def get_data_loader(local_rank, batch_size=8):
    dataset = VOCDetection(
        root="/path/to/VOCdevkit",
        year="2012",
        image_set="train",
        download=True,
        transforms=VOCTransform()
    )

    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn,
        sampler=sampler
    )
    return dataloader

# 4. 训练逻辑
def train_one_epoch(model, optimizer, dataloader, device,local_rank,epoch,scaler):
    model.train()
    running_loss = 0.0
    if local_rank == 0:
        pbar = tqdm(dataloader, desc=f"Rank {local_rank} Training Epoch {epoch}", position=0, ncols=100)
    else:
        pbar = dataloader
    i= 0 
    for images, targets in pbar:
        i+=1
        # 将所有图像和目标移动到设备
        optimizer.zero_grad()

        with autocast():
            losses = model(images, targets)
            loss = sum(loss for loss in losses.values())
            
        scaler.scale(loss).backward()
        
        scaler.step(optimizer)

        scaler.update()

        running_loss += loss.item()
       
        if local_rank == 0:
            pbar.set_postfix(loss=running_loss / len(dataloader))
        
    barrier()

    if local_rank == 0:
        pbar.close()

def train(local_rank):
    # 设置分布式环境
    device = setup_distributed()

    # 加载模型
    num_classes = 21  # VOC 2012 有 20 个类别 + 背景
    model = get_model(num_classes).to(device)
    model = DDP(model, device_ids=[local_rank])

    # 加载数据
    dataloader = get_data_loader(local_rank)


    # 优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    scaler = GradScaler()
    # 训练循环
    num_epochs = 10
    for epoch in range(num_epochs):
        # 分布式采样器需要设置 epoch
        start_time = time.time()
        dataloader.sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, dataloader, device,local_rank,epoch,scaler)
        
        if dist.get_rank() == 0:  # 仅主进程打印日志
            print(f"Epoch {epoch + 1} completed Time:{start_time-time.time():.2f}s")

    # 销毁分布式环境
    dist.destroy_process_group()

# 5. 主入口
if __name__ == "__main__":
    print("cuda:",int(os.environ["LOCAL_RANK"]))
    train(int(os.environ["LOCAL_RANK"]))
