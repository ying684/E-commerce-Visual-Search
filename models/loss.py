#  models/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features # Số lượng mặt hàng (8811 class)
        self.s = s                       # Scale factor
        self.m = m                       # Margin góc (Angular margin)
        
        # Ma trận trọng số (Weight) đại diện cho các tâm (center) của từng class
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # 1. Tính Cosine Similarity giữa Vector đặc trưng và Vector trọng số
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        
        # 2. Tính Sine
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2)).clamp(0, 1)
        
        # 3. Cộng Margin vào góc của nhãn đúng (Cos(theta + m))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # 4. Áp dụng Margin chỉ cho các label tương ứng
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        # Trả về output để đưa vào nn.CrossEntropyLoss()
        return output