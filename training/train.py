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
from preprocessing.real_world_dataset import CrossDomainTripletDataset

# Tăng batch_size lên 64 để tận dụng tối đa 2 GPU

def train_model(epochs=5, batch_size=64, learning_rate=1e-4, resume=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Đang tải dữ liệu thực tế (Shopee Dataset)...")
    csv_path = '../input/shopee-product-matching/train.csv' # Đường dẫn mặc định của Kaggle
    img_dir = '../input/shopee-product-matching/train_images/'
    
    train_dataset = CrossDomainTripletDataset(csv_file=csv_path, img_dir=img_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    model = CBIRBackbone(model_name='resnet50', embedding_dim=256)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    criterion = TripletMarginLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    os.makedirs('outputs', exist_ok=True)
    
    start_epoch = 0
    best_loss = float('inf')
    checkpoint_path = 'outputs/checkpoint_latest.pth'

    # CƠ CHẾ RESUME TRAINING
    if resume and os.path.exists(checkpoint_path):
        print(f"[*] Tìm thấy file checkpoint. Đang khôi phục trạng thái huấn luyện...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Phục hồi Model
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
            
        # Phục hồi Optimizer và các thông số
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        print(f"[+] Đã phục hồi thành công! Tiếp tục huấn luyện từ Epoch {start_epoch + 1}")

    print("Đang khởi động quá trình huấn luyện...")
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (anchor, positive, negative, _) in enumerate(train_loader):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            optimizer.zero_grad()
            emb_anchor, emb_positive, emb_negative = model(anchor), model(positive), model(negative)
            loss, _, _ = criterion(emb_anchor, emb_positive, emb_negative)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx}/{len(train_loader)}] | Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        print(f"==> Kết thúc Epoch {epoch+1} | Average Loss: {epoch_loss:.4f}")

        # LƯU CHECKPOINT SAU MỖI EPOCH (Để có thể resume)
        state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save({
            'epoch': epoch,
            'model_state_dict': state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss
        }, checkpoint_path)

        # LƯU MODEL TỐT NHẤT (Dùng cho Inference sau này)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(state_dict, 'outputs/best_cbir_model.pth')
            print(">>> Đã cập nhật mô hình tốt nhất (best_cbir_model.pth)\n")

if __name__ == "__main__":
    # Đổi resume=True để kích hoạt tính năng đọc lại checkpoint nếu có
    train_model(epochs=10, resume=True)