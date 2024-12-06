# -*- coding: utf-8 -*-
# @Time    : 2024/11/28 下午5:43
# @Author  : ysj
# @FileName: Faster-R-CNN.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/ydscc?type=blog

import torchvision
from torchvision import models

def get_model():

    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    num_classes =21

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    return model


