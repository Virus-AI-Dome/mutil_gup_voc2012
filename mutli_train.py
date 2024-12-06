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
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import xml.etree.ElementTree as ET
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cv2
from torchvision import models
# 类别名称到索引的映射
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import time
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

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

        return image, annotation,image_id
    
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

def get_voc_dataset(root, split, year="2012"):
    transform = transforms.Compose([transforms.ToTensor(),])
    dataset = VOCDetectionDataset(
    root=root, 
    year="2012", 
    image_set=split, 
    transforms=transform
)
    return dataset


def collate_fn(batch):
    return tuple(zip(*batch))

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


# def get_data_loader(local_rank, batch_size=8):
#     dataset = VOCDetection(
#         root="/path/to/VOCdevkit",
#         year="2012",
#         image_set="train",
#         download=True,
#         transforms=VOCTransform()
#     )

#     sampler = DistributedSampler(dataset)
#     dataloader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=(sampler is None),
#         num_workers=8,
#         pin_memory=True,
#         collate_fn=collate_fn,
#         sampler=sampler
#     )
#     return dataloader

# 验证函数
def calculate_iou(box1, box2):
    # 计算两个框的交集和并集
    x1, y1, x2, y2 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # 交集
    xi1 = max(x1, x1_2)
    yi1 = max(y1, y1_2)
    xi2 = min(x2, x2_2)
    yi2 = min(y2, y2_2)

    intersection = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # 并集
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    union = area1 + area2 - intersection

    iou = intersection / union if union != 0 else 0
    return iou

def evaluate(model, dataloader,local_rank):
    '''
    使用 evaluate 函数计算 mAP 以评估模型性能, 目标检测准确率计算MAP 
    '''
    model.eval()  # 设置为评估模式
    correct_predictions = 0
    total_predictions = 0
    total_gt_boxes = 0
    pbar = tqdm(dataloader, desc=f"Rank:{local_rank},Evaluate Epoch", position=0, ncols=100)
    for images, targets,image_id in tqdm(dataloader,desc="Training", ncols=100):
        images = [image.to(local_rank) for image in images]
        targets = [{k: v.to(local_rank) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            # 获取预测结果
            prediction = model(images)

        for i in range(len(images)):
            target = targets[i]
            prediction_result = prediction[i]
          
            gt_bboxes = target['boxes']
            gt_labels =  target['labels']
            
            pred_boxes = prediction_result['boxes'].cpu().numpy()
            pred_labels = prediction_result['labels'].cpu().numpy()
            pred_scores = prediction_result['scores'].cpu().numpy()

            # 根据 IoU 阈值计算准确率
            for gt_bbox, gt_label in zip(gt_bboxes, gt_labels):
                total_gt_boxes += 1
                best_iou = 0
                best_pred_idx = -1
                for idx, pred_bbox in enumerate(pred_boxes):
                    iou = calculate_iou(gt_bbox, pred_bbox)
                    if iou > best_iou and pred_labels[idx] == gt_label:
                        best_iou = iou
                        best_pred_idx = idx

                # 如果最佳 IoU > threshold，则视为一个正确预测
                if best_iou > 0.5 and best_pred_idx != -1:
                    correct_predictions += 1
                total_predictions += 1
        if local_rank == 0:
              pbar.set_postfix(val= correct_predictions / total_predictions)
              
    accuracy = correct_predictions / total_predictions if total_predictions != 0 else 0

    return accuracy


#5.  保存最优模型
def save_model(model,path,epoch,best_map,current_map):
    if current_map >best_map:
        print(f"Saving best model at epoch {epoch},Map: {current_map}")
        torch.save(model.state_dict(),path)
        return current_map
    return best_map



# 4. 训练逻辑
def train_one_epoch(model, optimizer, dataloader, device,local_rank,epoch,scaler):
    model.train()
    running_loss = 0.0
    if local_rank == 0:
        pbar = tqdm(dataloader, desc=f"Rank:{local_rank},Training Epoch {epoch}", position=0, ncols=100)
    else:
        pbar = dataloader
    i= 0 
    for images, targets,image_id in pbar:
        i+=1
        # 将所有图像和目标移动到设备
        images = [image.to(local_rank) for image in images]
        targets = [{k: v.to(local_rank) for k, v in t.items()} for t in targets]
    
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

        torch.cuda.empty_cache()

def train(args,local_rank):
    # 设置分布式环境
    device = setup_distributed()
    # 加载模型
    num_classes = 21  # VOC 2012 有 20 个类别 + 背景
    model = get_model(num_classes).to(device)
    model = DDP(model, device_ids=[local_rank])

    # 加载训练集voc数据
    train_dataset = get_voc_dataset(args.data_path,'train')
    #使用DataLoader 加载voc数据
    train_dataloader = get_dataloader(train_dataset, args.batch_size, args.num_workers, distributed=True)
    
    #加载val 数据集
    val_dataset = get_voc_dataset(args.data_path,'val')
    val_dataloader = get_dataloader(val_dataset,args.batch_size,args.num_workers,distributed =True)

    
    # 优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    scaler = GradScaler()
    # 训练循环
    best_map = float('-inf')
    for epoch in range(args.epochs):
        # 分布式采样器需要设置 epoch
        start_time = time.time()
        train_dataloader.sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, train_dataloader, device,local_rank,epoch,scaler)
        dist.barrier() 
        current_map = evaluate(model,val_dataloader,local_rank)
  
        if dist.get_rank() == 0:  # 仅主进程打印日志
            best_map = save_model(model.module, "best_faster_rcnn.pth", epoch, best_map, current_map)
            print(f"Epoch {epoch + 1} completed Time:{start_time-time.time():.2f}s, current_map:{current_map}")

    # 销毁分布式环境
    dist.destroy_process_group()

# 5. 主入口
if __name__ == "__main__":
    # torchrun --standalone --nnodes=1 --nproc_per_node=4 torchrun_demo.py

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",type=str, default='/ydemo_ai/ImgNet/VOCdevkit',help='Path to VOC2012 root')
    parser.add_argument("--batch_size",type=int,default=8,help='Batch size per GPU')
    parser.add_argument("--num_workers",type=int,default=48,help='Number of data loader worker')
    parser.add_argument("--epochs",type=int,default=10, help='Number of Training epochs')

    args = parser.parse_args()
    print("cuda:",int(os.environ["LOCAL_RANK"]))
    train(args,int(os.environ["LOCAL_RANK"]))

    
