import torch
import torch.nn as nn
from torchvision import models

class CBIRBackbone(nn.Module):
    def __init__(self, model_name='resnet50', embedding_dim=256):
        super(CBIRBackbone, self).__init__()
        self.model_name = model_name
        
        if model_name == 'resnet50':
            # Tải ResNet50 pre-trained trên ImageNet
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            # Thay thế lớp Fully Connected (fc) cuối cùng
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, embedding_dim)
            
        elif model_name == 'mobilenet_v3_large':
            # Tải MobileNetV3Large pre-trained
            self.backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
            # Thay thế lớp Classifier cuối cùng
            in_features = self.backbone.classifier[3].in_features
            self.backbone.classifier[3] = nn.Linear(in_features, embedding_dim)
            
        else:
            raise ValueError("Hiện chỉ hỗ trợ 'resnet50' hoặc 'mobilenet_v3_large'")

    def forward(self, x):
        # Truyền ảnh qua backbone để lấy đặc trưng
        features = self.backbone(x)
        
        # Áp dụng chuẩn hóa L2 trên dimension 1
        # Ép các vector nằm trên bề mặt một siêu cầu (hypersphere) bán kính = 1
        embeddings = nn.functional.normalize(features, p=2, dim=1)
        return embeddings