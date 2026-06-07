import torch
from preprocessing.triplet_dataset import FashionMNISTTriplet
from models.backbone import CBIRBackbone
from models.loss import TripletMarginLoss

def test_pipeline():
    print("1. Đang tải Data...")
    train_dataset = FashionMNISTTriplet(train=True)
    anchor, positive, negative, label = train_dataset[0]
    
    # Giả lập Batch Size = 2 bằng cách lặp lại tensor (để test tính toán mean của loss)
    anchor_batch = torch.stack([anchor, anchor])
    positive_batch = torch.stack([positive, positive])
    negative_batch = torch.stack([negative, negative])
    
    print("2. Khởi tạo Backbone và Loss Function...")
    model = CBIRBackbone(model_name='resnet50', embedding_dim=256)
    model.eval()
    
    # Margin = 1.0 (Heuristic điều chỉnh thực nghiệm)
    criterion = TripletMarginLoss(margin=1.0) 
    
    print("3. Forward Pass toàn bộ Triplet...")
    with torch.no_grad():
        emb_anchor = model(anchor_batch)
        emb_positive = model(positive_batch)
        emb_negative = model(negative_batch)
        
        loss, d_pos, d_neg = criterion(emb_anchor, emb_positive, emb_negative)
        
    print(f"\n=> Khoảng cách trung bình Anchor-Positive: {d_pos.item():.4f}")
    print(f"=> Khoảng cách trung bình Anchor-Negative: {d_neg.item():.4f}")
    print(f"=> Giá trị Triplet Loss: {loss.item():.4f}")
    
    print("\n[THÀNH CÔNG] Pipeline hoàn chỉnh. Sẵn sàng Backpropagation!")

if __name__ == "__main__":
    test_pipeline()