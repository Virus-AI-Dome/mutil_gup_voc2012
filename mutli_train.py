import torch.optim as optim
import torch 
import torch.nn as nn 
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel  import DistributedDataParallel as DDP
import os
import time
from Vocdataset import VocDataset
import torchvision.transforms as T
from torchvision import models
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
import torch.distributed as dist
from datetime import timedelta


# 创建分布式环境
def setup(rank,world_size):
    os.environ['NCCL_DEBUG'] = 'INFO'
    # os.environ['NCCL_ALGO'] = 'Tree'
    os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['CCL_IB_DISABLE'] = '1'
    os.environ['NCCL_SHM_DISABLE'] ='1'
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = '12355'
    # os.environ["NCCL_DEBUG_SUBSYS"] = 'ALL'
    # os.environ[' NCCL_NTHREADS'] = '4'
    # os.environ[' NCCL_NET_GDR_LEVEL'] = '2'
    os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'
    os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] ='1'

    dist.init_process_group('nccl', rank=rank, world_size=world_size,
    init_method="env://",
    timeout=timedelta(minutes=10))
    torch.cuda.set_device(rank)

def get_model():

    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    num_classes =21

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    return model
#销毁分布式环境

def cleanup():
    dist.destroy_process_group()


def get_dataloader(batch_size, rank, world_size):

    # 数据转换
    transform = T.Compose([T.ToTensor()])

    # 加载数据
    train_dataset = VocDataset(root_dir=r'/ysj_demo/ImgNet/VOCdevkit/VOC2012/',
                               transform=transform, split='trainval')

    sampler = DistributedSampler(train_dataset,num_replicas=world_size, rank=rank, shuffle=False)

    dataloader = DataLoader(train_dataset,batch_size= batch_size, sampler= sampler,num_workers=8,pin_memory=True)

    return dataloader

def train(rank,world_size,epochs= 200,batch_size=32):

    setup(rank,world_size)

    device = torch.device(f'cuda:{rank}')
     
    dataloader = get_dataloader(batch_size, rank, world_size)

    model = get_model()
   
    model = model.to(rank)

    # params = [p for p in model.parameters() if p.requires_grad]
    # for param in model.backbone.parameters():
    #     param.requires_grad = False

    ddp_model = DDP(model,device_ids = [rank])

    torch.nn.parallel.DistributedDataParallel(ddp_model,gradient_as_bucket_view =True)

    criterion = nn.MSELoss()
   
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)

    for epoch in range(epochs):
        start_t = time.time()
        ddp_model.train()
        total_loss = 0.0
        for images, boxes, labels in tqdm(dataloader, desc="Training", ncols=100):
            images = [image.to(rank) for image in images]
            boxes = [box.to(rank) for box in boxes]
            labels = [label.to(rank) for label in labels]

            loss_dict = ddp_model(images, targets=[{'boxes': box, 'labels': label} for box, label in zip(boxes, labels)])
            print("loss_dict:",loss_dict)
            # Compute total loss
            loss = criterion(loss_dict, target)    # Backward pass
            losses.backward()
            optimizer.step()
            running_loss += losses.item()
            
        if rank == 0:  # 只在主进程打印日志
            print(f"batch time:{(time.time() - start_t):.4f}")
            print(f"Epoch {epoch + 1}/{epochs},Loss:{running_loss / len(dataloader):.4f}")

    if rank == 0:
        torch.save(ddp_model.state_dict(), "ddp_model.pth")
        print("Model Save")
    # 注销分布式环境
    cleanup()


def main():

    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUS  for training")
    torch.multiprocessing.spawn(train, args=(4,), nprocs=world_size,join=True)



if __name__ == "__main__":
    main()








    

   







