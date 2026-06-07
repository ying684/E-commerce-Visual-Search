# File: evaluation/evaluate.py

import sys
import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import faiss  # Thư viện lõi cho Vector Search

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.backbone import CBIRBackbone

class ShopeeInferenceDataset(Dataset):
    """Dataset chuyên dụng cho khâu truy xuất (Không Augmentation)"""
    def __init__(self, csv_file, img_dir):
        self.data_df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.image_paths = self.data_df['image'].values
        self.labels = self.data_df['label_group'].values
        self.posting_ids = self.data_df['posting_id'].values
        
        # Chỉ giữ lại Transform tĩnh để bảo toàn thông tin gốc
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        img_name = self.image_paths[index]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, self.labels[index], self.posting_ids[index]

def extract_embeddings(dataloader, model, device):
    """Chạy mô hình để biến toàn bộ kho ảnh thành Vector"""
    model.eval()
    all_embeddings = []
    all_labels = []
    all_postings = []
    
    print("Đang trích xuất Vector cho Database... (Có thể mất vài phút)")
    with torch.no_grad():
        for imgs, labels, postings in dataloader:
            imgs = imgs.to(device)
            # Ép kiểu float32 để tương thích tuyệt đối với FAISS C++ backend
            embeddings = model(imgs).cpu().numpy().astype('float32') 
            all_embeddings.append(embeddings)
            all_labels.extend(labels.numpy())
            all_postings.extend(postings)
            
    return np.vstack(all_embeddings), np.array(all_labels), np.array(all_postings)

def main():
    # Khai báo cấu hình Kaggle
    csv_path = '/kaggle/input/competitions/shopee-product-matching/train.csv'
    img_dir = '/kaggle/input/competitions/shopee-product-matching/train_images/'
    model_path = '/kaggle/working/E-commerce-Visual-Search/outputs/best_cbir_model.pth'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Tải Model Core
    print("1. Khởi tạo AI Core và nạp trọng số...")
    model = CBIRBackbone(model_name='resnet50', embedding_dim=256)
    
    # Load state dict chuẩn xác (Xử lý trường hợp model lưu bằng DataParallel)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    
    # 2. Tải Dữ liệu
    print("2. Đang chuẩn bị DataLoader Inference...")
    dataset = ShopeeInferenceDataset(csv_path, img_dir)
    # Tăng batch_size khi inference vì không tốn VRAM cho Gradient tính toán
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)
    
    # 3. Trích xuất Embeddings
    embeddings, labels, postings = extract_embeddings(dataloader, model, device)
    print(f"Hoàn tất! Kích thước Latent Space: {embeddings.shape}")
    
    # 4. KHỞI TẠO FAISS VECTOR DATABASE
    print("\n3. Đang lập chỉ mục bằng FAISS (IndexFlatL2)...")
    dimension = embeddings.shape[1]
    
    # Sử dụng IndexFlatL2 cho độ chính xác tuyệt đối (Exhaustive Search)
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)
    faiss.write_index(faiss_index, '/kaggle/working/E-commerce-Visual-Search/outputs/faiss_database.index')
    print("Đã lưu FAISS Index xuống ổ cứng (faiss_database.index)")

    print(f"Đã nạp {faiss_index.ntotal} vector vào không gian tìm kiếm.")
    
    # 5. MÔ PHỎNG TRUY VẤN (QUERY) VÀ TÍNH RECALL
    # Lấy 1000 ảnh đầu tiên làm Query để đánh giá hiệu năng thuật toán
    query_embeddings = embeddings[:1000]
    query_labels = labels[:1000]
    
    k_neighbors = 5
    print(f"\n4. Bắt đầu tìm kiếm {k_neighbors} Láng giềng gần nhất cho 1000 Query...")
    # FAISS trả về ma trận khoảng cách (D) và chỉ số index của ảnh (I)
    D, I = faiss_index.search(query_embeddings, k_neighbors)
    
    correct_retrievals = 0
    for i in range(len(query_labels)):
        q_label = query_labels[i]
        # Lấy nhãn của các ảnh được FAISS tìm thấy
        retrieved_labels = labels[I[i]] 
        
        # Bỏ qua chính nó (vị trí đầu tiên luôn là chính bức ảnh đó vì khoảng cách = 0)
        # Nếu có ít nhất 1 ảnh cùng nhãn trong top K -> Tính là tìm kiếm thành công
        if q_label in retrieved_labels[1:]:
            correct_retrievals += 1
            
    recall_at_k = correct_retrievals / len(query_labels)
    print("="*40)
    print("KẾT QUẢ ĐÁNH GIÁ (EVALUATION METRICS)")
    print("="*40)
    print(f"Recall@{k_neighbors} : {recall_at_k * 100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    main()