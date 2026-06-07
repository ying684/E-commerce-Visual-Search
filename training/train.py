import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from preprocessing.triplet_dataset import FashionMNISTTriplet
from models.backbone import CBIRBackbone
from models.loss import TripletMarginLoss

def train_model(epochs=5, batch_size=32, learning_rate=1e-4):
    # Tự động nhận diện GPU (CUDA) nếu có, ngược lại dùng CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang khởi động quá trình huấn luyện trên thiết bị: {device}")

    # 1. Khởi tạo DataLoader
    print("Đang tải dữ liệu...")
    train_dataset = FashionMNISTTriplet(train=True)
    # Tăng num_workers để CPU nạp dữ liệu nhanh hơn cho GPU
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # 2. Khởi tạo Model, Loss, và Optimizer
    model = CBIRBackbone(model_name='resnet50', embedding_dim=256).to(device)
    # Heuristic điều chỉnh thực nghiệm (Margin = 1.0)
    criterion = TripletMarginLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    os.makedirs('outputs', exist_ok=True)
    best_loss = float('inf')

    # 3. Vòng lặp huấn luyện chính
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (anchor, positive, negative, _) in enumerate(train_loader):
            # Đưa dữ liệu lên GPU/CPU
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            # Reset gradient
            optimizer.zero_grad()

            # Forward pass
            emb_anchor = model(anchor)
            emb_positive = model(positive)
            emb_negative = model(negative)

            # Tính loss
            loss, _, _ = criterion(emb_anchor, emb_positive, emb_negative)

            # Backpropagation & Optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # In log mỗi 50 batches
            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx}/{len(train_loader)}] | Loss: {loss.item():.4f}")

        # Đánh giá cuối Epoch
        epoch_loss = running_loss / len(train_loader)
        print(f"==> Kết thúc Epoch {epoch+1} | Average Loss: {epoch_loss:.4f}")

        # Lưu lại file trọng số tốt nhất để bảo toàn thành quả thực nghiệm
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), 'outputs/best_cbir_model.pth')
            print(">>> Đã cập nhật và lưu mô hình tốt nhất tại outputs/best_cbir_model.pth\n")

if __name__ == "__main__":
    # Bắt đầu train với 5 epochs để test độ hội tụ trên Kaggle
    train_model(epochs=5)