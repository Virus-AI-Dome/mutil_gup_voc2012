import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.transforms import functional as F
from PIL import Image
import cv2
import matplotlib.pyplot as plt
# 定义模型
num_classes = 21  # VOC 2012 包括 20 类目标 + 背景
backbone = resnet_fpn_backbone('resnet50', pretrained=False)  # 不加载预训练权重
model = FasterRCNN(backbone, num_classes=num_classes)

# 加载训练好的权重
weight_path = "/ydemo_ai/ImgNet/Faster-R-CNN/best_faster_rcnn.pth"  # 替换为你的权重文件路径
model.load_state_dict(torch.load(weight_path, map_location='cpu'))
model.eval()  # 切换为推理模式

# 使用 GPU（如果可用）
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
#加载图像
img_path = '/ydemo_ai/ImgNet/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg'
image = Image.open(img_path).convert('RGB')

img_tensor = F.to_tensor(image).unsqueeze(0).to(device)


# 模型推理
with torch.no_grad():
    predictions =  model(img_tensor)

#计算所有标签预测值
boxes = predictions[0]['boxes'].cpu().numpy()
labels = predictions[0]['labels'].cpu().numpy()
scores = predictions[0]['scores'].cpu().numpy()

#筛选置信大于阈值的结果
threshold = 0.7
filtered_indices = scores > threshold
boxes = boxes[filtered_indices]
labels = labels[filtered_indices]
scores = scores[filtered_indices]

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]


image_np = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
for box, label, score in zip(boxes, labels, scores):
    x1, y1, x2, y2 = box
    cv2.rectangle(image_np, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    cv2.putText(
        image_np, f"{VOC_CLASSES[label]}: {score:.2f}",
        (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
    )

# 显示结果
plt.imshow(image_np)
plt.axis("off")
output_path = "output_image.png"
plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
print(f"图像已保存至 {output_path}")


