# backbone.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# 1. ĐỊNH NGHĨA GEM POOLING (Vũ khí bí mật của CBIR)
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        # Biến p là tham số có thể học được (learnable) trong lúc train
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # clamp để tránh lỗi chia cho 0 hoặc log(0)
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

# 2. KIẾN TRÚC MẠNG CHÍNH
class CBIRBackbone(nn.Module):
    def __init__(self, model_name='resnet50', embedding_dim=256):
        super(CBIRBackbone, self).__init__()
        
        # Load pre-trained ResNet50
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Cắt bỏ 2 lớp cuối mặc định của ResNet (AvgPool và Fully Connected)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # Lắp GeM Pooling vào thay thế
        self.pool = GeM()
        
        # Cấu trúc Neck: Tối ưu không gian vector trước khi tính khoảng cách
        self.neck = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        x = self.features(x)         # Output shape: [Batch, 2048, 7, 7]
        x = self.pool(x)             # Output shape: [Batch, 2048, 1, 1]
        x = x.view(x.size(0), -1)    # Flatten -> [Batch, 2048]
        x = self.neck(x)             # Output shape: [Batch, 256]
        
        # Chuẩn hóa L2: Ép tất cả các vector lên bề mặt hình cầu bán kính 1
        x = F.normalize(x, p=2, dim=1)
        return x