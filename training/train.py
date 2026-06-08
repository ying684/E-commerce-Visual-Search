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
from torch.cuda.amp import autocast, GradScaler

# Import các vũ khí hạng nặng chúng ta vừa chế tạo
from preprocessing.real_world_dataset import ShopeeArcFaceDataset
from models.backbone import CBIRBackbone
from models.loss import ArcFaceLoss
from evaluation.evaluate import evaluate_model
from utils.logger import TrainingLogger

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN (Dễ dàng đổi khi lên Kaggle)
# ==========================================
TRAIN_CSV = '/kaggle/input/datasets/nguynanhnhnl/spilit/train_split.csv'
VAL_CSV = '/kaggle/input/datasets/nguynanhnhnl/spilit/val_split.csv'
IMG_DIR = '/kaggle/input/competitions/shopee-product-matching/train_images'

def train_model(epochs=10, batch_size=64, learning_rate=3e-4, resume=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Đang chạy trên thiết bị: {device}")

    # 1. Chuẩn bị Dữ liệu
    print("[*] Đang tải dữ liệu...")
    train_dataset = ShopeeArcFaceDataset(csv_file=TRAIN_CSV, img_dir=IMG_DIR)
    val_dataset = ShopeeArcFaceDataset(csv_file=VAL_CSV, img_dir=IMG_DIR)
    
    num_classes = train_dataset.num_classes
    print(f"[*] Số lượng mặt hàng (Classes) dùng để Train: {num_classes}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4 if torch.cuda.is_available() else 0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4 if torch.cuda.is_available() else 0)

    # 2. Khởi tạo AI Core & ArcFace
    model = CBIRBackbone(model_name='resnet50', embedding_dim=256)
    if torch.cuda.device_count() > 1:
        print(f"[*] Kích hoạt DataParallel trên {torch.cuda.device_count()} GPUs...")
        model = nn.DataParallel(model)
    model = model.to(device)

    arcface = ArcFaceLoss(in_features=256, out_features=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    # 3. Tối ưu hóa
    optimizer = optim.AdamW([
        {'params': model.parameters()},
        {'params': arcface.parameters()}
    ], lr=learning_rate, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()
    
    # 4. CƠ CHẾ KHÔI PHỤC (RESUME & CHECKPOINTING)
    os.makedirs('outputs', exist_ok=True)
    logger = TrainingLogger(save_path='outputs/training_history.csv')
    checkpoint_path = 'outputs/checkpoint_latest.pth'
    
    start_epoch = 0
    best_map = 0.0

    if resume and os.path.exists(checkpoint_path):
        print(f"\n[!] TÌM THẤY CHECKPOINT. Đang khôi phục từ: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load Model
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
            
        # Load ArcFace, Optimizer, Scheduler
        arcface.load_state_dict(checkpoint['arcface_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_map = checkpoint['best_map']
        print(f"[!] Đã khôi phục thành công! Tiếp tục train từ Epoch {start_epoch + 1}...\n")

    print("\n" + "="*50)
    print(" BẮT ĐẦU PIPELINE HUẤN LUYỆN (ARCFACE + GEM POOLING)")
    print("="*50)

    # Vòng lặp bắt đầu từ start_epoch
    for epoch in range(start_epoch, epochs):
        model.train()
        arcface.train()
        running_loss = 0.0

        for batch_idx, (images, labels, _, _) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast():
                embeddings = model(images)
                arcface_outputs = arcface(embeddings, labels)
                loss = criterion(arcface_outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx}/{len(train_loader)}] | Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\n==> Đang làm bài Thi Cuối Kỳ (Validation) cho Epoch {epoch+1}...")
        val_map = evaluate_model(model, val_loader, device, k=5)
        
        print(f"KẾT QUẢ EPOCH {epoch+1}:")
        print(f" - Train Loss : {epoch_loss:.4f}")
        print(f" - Val mAP@5  : {val_map:.4f}")
        print(f" - L.Rate     : {current_lr:.6f}\n")

        logger.log(epoch+1, epoch_loss, val_map, current_lr)

        # GHI CHECKPOINT LIÊN TỤC SAU MỖI EPOCH
        state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save({
            'epoch': epoch,
            'model_state_dict': state_dict,
            'arcface_state_dict': arcface.state_dict(), # Cực kỳ quan trọng
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_map': best_map
        }, checkpoint_path)

        if val_map > best_map:
            best_map = val_map
            torch.save(state_dict, 'outputs/best_cbir_model.pth')
            print(">>> 🎯 Đã cập nhật mô hình Đỉnh cao mới (best_cbir_model.pth)\n")

        scheduler.step()

if __name__ == "__main__":
    train_model(epochs=10, resume=True) # Mặc định bật Resume