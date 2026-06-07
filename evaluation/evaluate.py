import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.backbone import CBIRBackbone

def extract_features(dataloader, model, device):
    """Quét toàn bộ ảnh trong dataloader để trích xuất vector embeddings"""
    model.eval()
    all_embeddings = []
    all_labels = []
    
    print("Đang trích xuất Vector Embeddings...")
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            embeddings = model(imgs)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())
            
    return np.vstack(all_embeddings), np.concatenate(all_labels)

def calculate_recall_at_k(query_embeddings, query_labels, gallery_embeddings, gallery_labels, k=5):
    """Tính toán chỉ số Recall@K bằng ma trận khoảng cách Euclide"""
    print(f"Đang tìm kiếm KNN và tính toán Recall@{k}...")
    
    # Chuyển numpy array về tensor để PyTorch tính toán khoảng cách bằng C++ backend cho nhanh
    query_tensor = torch.tensor(query_embeddings)
    gallery_tensor = torch.tensor(gallery_embeddings)
    
    # Tính ma trận khoảng cách đôi một (Pairwise Distance) giữa Query và Gallery
    distances = torch.cdist(query_tensor, gallery_tensor, p=2)
    
    # Lấy ra K index có khoảng cách nhỏ nhất (K ảnh giống nhất)
    # distance.topk trả về giá trị lớn nhất, nên ta truyền vào -distances để lấy giá trị nhỏ nhất
    _, topk_indices = (-distances).topk(k, dim=1)
    
    topk_indices = topk_indices.numpy()
    correct = 0
    
    for i, query_label in enumerate(query_labels):
        # Lấy nhãn của K ảnh kết quả
        retrieved_labels = gallery_labels[topk_indices[i]]
        
        # Nếu trong K ảnh trả về có ít nhất 1 ảnh cùng nhãn với Query -> Tính là 1 lần Recall thành công
        if query_label in retrieved_labels:
            correct += 1
            
    recall = correct / len(query_labels)
    return recall

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Tải Model Weights đã train
    model = CBIRBackbone(model_name='resnet50', embedding_dim=256)
    model_path = 'outputs/best_cbir_model.pth'
    
    if not os.path.exists(model_path):
        print("Không tìm thấy model weight! Hãy chắc chắn file best_cbir_model.pth nằm trong thư mục outputs.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    print("Đã tải thành công weights mô hình AI!")

    # 2. Chuẩn bị tập dữ liệu Test
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    # 3. Trích xuất đặc trưng cho toàn bộ 10,000 ảnh Test
    embeddings, labels = extract_features(test_loader, model, device)
    print(f"Hoàn tất! Kích thước kho dữ liệu Vector: {embeddings.shape}")

    # 4. Giả lập: Lấy 500 ảnh đầu tiên làm Query, 9500 ảnh còn lại làm Gallery (Database)
    query_emb, query_lbl = embeddings[:500], labels[:500]
    gallery_emb, gallery_lbl = embeddings[500:], labels[500:]

    # 5. Đánh giá thuật toán
    recall_at_5 = calculate_recall_at_k(query_emb, query_lbl, gallery_emb, gallery_lbl, k=5)
    recall_at_10 = calculate_recall_at_k(query_emb, query_lbl, gallery_emb, gallery_lbl, k=10)
    
    print("\n" + "="*40)
    print("KẾT QUẢ ĐÁNH GIÁ (EVALUATION METRICS)")
    print("="*40)
    print(f"Recall@5 : {recall_at_5 * 100:.2f}%")
    print(f"Recall@10: {recall_at_10 * 100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    main()