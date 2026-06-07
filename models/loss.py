import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletMarginLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletMarginLoss, self).__init__()
        # Khởi tạo Heuristic điều chỉnh thực nghiệm
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Tính khoảng cách L2 giữa Anchor - Positive
        distance_positive = F.pairwise_distance(anchor, positive, p=2)
        
        # Tính khoảng cách L2 giữa Anchor - Negative
        distance_negative = F.pairwise_distance(anchor, negative, p=2)
        
        # Tính toán giá trị Loss dựa trên margin
        losses = F.relu(distance_positive - distance_negative + self.margin)
        
        # Trả về giá trị trung bình của toàn bộ batch
        return losses.mean(), distance_positive.mean(), distance_negative.mean()