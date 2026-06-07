# File: training/train.py

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler # <--- Import AMP

from preprocessing.real_world_dataset import ShopeeRawTripletDataset
from models.backbone import CBIRBackbone
from models.loss import TripletMarginLoss

# Tăng batch_size lên 128 vì đã có AMP giải phóng VRAM
def train_model(epochs=10, batch_size=128, learning_rate=1e-4, resume=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Đang tải Dữ liệu thô Shopee Price Match Guarantee...")
    csv_path = '/kaggle/input/shopee-product-matching/train.csv'
    img_dir = '/kaggle/input/shopee-product-matching/train_images/'

    train_dataset = ShopeeRawTripletDataset(csv_file=csv_path, img_dir=img_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = CBIRBackbone(model_name='resnet50', embedding_dim=256)
    if torch.cuda.device_count() > 1:
        print(f"Kích hoạt DataParallel trên {torch.cuda.device_count()} GPUs...")
        model = nn.DataParallel(model)
    model = model.to(device)

    criterion = TripletMarginLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # NÂNG CẤP 2 & 3: Khởi tạo Bộ giảm LR và Scaler cho Mixed Precision
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    scaler = GradScaler() 

    os.makedirs('outputs', exist_ok=True)
    start_epoch = 0
    best_loss = float('inf')
    checkpoint_path = 'outputs/checkpoint_latest.pth'

    if resume and os.path.exists(checkpoint_path):
        print("[*] Khôi phục trạng thái huấn luyện từ Checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        scheduler.load_state_dict(checkpoint.get('scheduler_state_dict', scheduler.state_dict()))

    print("Bắt đầu Pipeline Huấn luyện AI Core (Tối ưu AMP)...")
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (anchor, positive, negative, _) in enumerate(train_loader):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            optimizer.zero_grad()

            # KÍCH HOẠT AMP (Ép chạy Float16 để tăng tốc)
            with autocast():
                emb_anchor, emb_positive, emb_negative = model(anchor), model(positive), model(negative)
                loss, _, _ = criterion(emb_anchor, emb_positive, emb_negative)
            
            # Scaler xử lý gradient chống tràn số lượng nhỏ
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx}/{len(train_loader)}] | Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        print(f"==> Kết thúc Epoch {epoch+1} | Average Loss: {epoch_loss:.4f}")
        
        # Cập nhật Scheduler
        scheduler.step(epoch_loss)

        state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save({
            'epoch': epoch,
            'model_state_dict': state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_loss': best_loss
        }, checkpoint_path)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(state_dict, 'outputs/best_cbir_model.pth')
            print(">>> Đã cập nhật mô hình tốt nhất (best_cbir_model.pth)\n")

if __name__ == "__main__":
    train_model(epochs=15, resume=True) # Train 15 Epochs vì ta có Scheduler điều phối