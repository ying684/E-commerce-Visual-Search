import os
import torch
import numpy as np
import pandas as pd
import faiss
from PIL import Image
from torchvision import transforms
from models.backbone import CBIRBackbone

class VisualSearchEngine:
    def __init__(self, model_path, csv_path, img_dir, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_dir = img_dir
        
        # 1. Khởi tạo Model
        print("[*] Đang tải AI Core...")
        self.model = CBIRBackbone(model_name='resnet50', embedding_dim=256)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()

        # 2. Khởi tạo Transform (Không Augmentation)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 3. Nạp Metadata
        print("[*] Đang tải cơ sở dữ liệu sản phẩm...")
        self.df = pd.read_csv(csv_path)
        self.image_paths = self.df['image'].values
        self.posting_ids = self.df['posting_id'].values

        # 4. Tự động Build Index hoặc nạp Index nếu có sẵn
        # Trong thực tế, ta lưu FAISS index ra file .index để không phải build lại mỗi lần
        self.index_path = 'outputs/faiss_database.index'
        if os.path.exists(self.index_path):
            print("[*] Đã tìm thấy FAISS Index, đang nạp lên RAM...")
            self.index = faiss.read_index(self.index_path)
        else:
            print("[!] Chưa có FAISS Index. Cần build index trước khi tìm kiếm (Chạy file evaluate.py và thêm hàm faiss.write_index).")
            self.index = None

    def _extract_vector(self, image_path):
        """Trích xuất 256-D vector từ 1 tấm ảnh duy nhất"""
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            vector = self.model(img_tensor).cpu().numpy().astype('float32')
        return vector

    def search(self, query_image_path, top_k=5):
        """Thực thi tìm kiếm và trả về kết quả chuẩn format"""
        if self.index is None:
            raise ValueError("FAISS Index chưa được khởi tạo!")

        # Bước 1: Nén ảnh thành vector
        query_vector = self._extract_vector(query_image_path)

        # Bước 2: Truy vấn FAISS (Tốc độ mili-giây)
        distances, indices = self.index.search(query_vector, top_k)

        # Bước 3: Đóng gói kết quả (JSON-like)
        results = []
        for i in range(top_k):
            idx = indices[0][i]
            dist = float(distances[0][i])
            results.append({
                "rank": i + 1,
                "posting_id": self.posting_ids[idx],
                "image_filename": self.image_paths[idx],
                "distance": dist,
                "image_url": os.path.join(self.img_dir, self.image_paths[idx])
            })
            
        return {
            "query_image": query_image_path,
            "top_k": top_k,
            "results": results
        }