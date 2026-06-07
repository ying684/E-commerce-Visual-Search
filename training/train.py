import sys
import os

# 1. FIX LỖI ĐƯỜNG DẪN: Lấy thư mục gốc (ecommerce_cbir) và ép vào sys.path
# Nhờ đoạn này, em có thể chạy thoải mái: python training/train.py
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from preprocessing.triplet_dataset import FashionMNISTTriplet
from models.backbone import CBIRBackbone
from models.loss import TripletMarginLoss

# Tăng batch_size lên 64 để tận dụng tối đa 2 GPU
def train_model(epochs=5, batch_size=64, learning_rate=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Đang khởi động pipeline huấn luyện...")

    # 1. Khởi tạo DataLoader
    print("Đang tải dữ liệu...")
    train_dataset = FashionMNISTTriplet(train=True)
    # Tăng num_workers lên 4 để CPU bơm data nhanh hơn cho 2 GPU
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 2. Khởi tạo Model
    model = CBIRBackbone(model_name='resnet50', embedding_dim=256)

    # BẬT CHẾ ĐỘ MULTI-GPU NẾU PHÁT HIỆN NHIỀU HƠN 1 CARD
    if torch.cuda.device_count() > 1:
        print(f"Tuyệt vời! Phát hiện {torch.cuda.device_count()} GPUs. Đang kích hoạt DataParallel...")
        # Lớp bọc này sẽ tự động chia nhỏ batch data gửi tới 2 card và gom gradient lại
        model = nn.DataParallel(model)

    # Đẩy mô hình lên device (GPU(s) hoặc CPU)
    model = model.to(device)

    criterion = TripletMarginLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    os.makedirs('outputs', exist_ok=True)
    best_loss = float('inf')

    # 3. Vòng lặp huấn luyện chính
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (anchor, positive, negative, _) in enumerate(train_loader):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            optimizer.zero_grad()

            emb_anchor = model(anchor)
            emb_positive = model(positive)
            emb_negative = model(negative)

            loss, _, _ = criterion(emb_anchor, emb_positive, emb_negative)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx}/{len(train_loader)}] | Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        print(f"==> Kết thúc Epoch {epoch+1} | Average Loss: {epoch_loss:.4f}")

        # LƯU MÔ HÌNH THÔNG MINH: Bóc tách DataParallel trước khi lưu
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            
            # Nếu model đang bọc trong DataParallel, lấy model.module
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, 'outputs/best_cbir_model.pth')
            print(">>> Đã cập nhật và lưu mô hình tốt nhất tại outputs/best_cbir_model.pth\n")

if __name__ == "__main__":
    train_model(epochs=5)