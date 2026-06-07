import torch
from preprocessing.triplet_dataset import FashionMNISTTriplet

def test_pipeline():
    print("Đang khởi tạo Triplet Dataset...")
    train_dataset = FashionMNISTTriplet(train=True)
    
    # Lấy thử mẫu đầu tiên
    anchor, positive, negative, label = train_dataset[0]
    
    print(f"Hoàn tất! Tổng số mẫu train: {len(train_dataset)}")
    print(f"Class của Anchor: {label}")
    print(f"Kích thước tensor (C, H, W): {anchor.shape}")
    
    # Kiểm tra kích thước xem đã chuẩn 3 kênh, 224x224 chưa
    assert anchor.shape == torch.Size([3, 224, 224]), "Sai kích thước Tensor!"
    print("=> Data Loader đã sẵn sàng cho Backbone CNN!")

if __name__ == "__main__":
    test_pipeline()