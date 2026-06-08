import os
import torch
import faiss
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# Import các module chuẩn xác từ kiến trúc của bạn
from models.backbone import CBIRBackbone
from core.config import settings
from utils.image_processing import process_image

class CBIRDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Lấy tên file từ CSV
        img_name = self.df['image'].iloc[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # SỬ DỤNG HÀM XỬ LÝ ẢNH CHUẨN CỦA BẠN (Lót nền trắng)
        img = process_image(img_path)
        
        if self.transform:
            img = self.transform(img)
            
        return img

def build_index():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Bắt đầu trích xuất đặc trưng trên: {device}")

    # 1. Khởi tạo AI Core (ArcFace + GeM)
    model = CBIRBackbone(model_name='resnet50', embedding_dim=256)
    model.load_state_dict(torch.load(settings.MODEL_PATH, map_location=device))
    model.to(device).eval()

    # 2. Tiền xử lý chuẩn ImageNet (Khớp với ResNet50_Weights.IMAGENET1K_V1)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. Load Dữ Liệu
    print(f"[INFO] Đọc metadata từ: {settings.CSV_PATH}")
    df = pd.read_csv(settings.CSV_PATH)
    
    # BẮT BUỘC: shuffle=False để giữ tính toàn vẹn thông tin với vị trí Index
    dataset = CBIRDataset(df, settings.IMG_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)

    embeddings = []

    # 4. Trích xuất Đặc trưng
    with torch.no_grad():
        for batch_imgs in tqdm(dataloader, desc="Đang trích xuất vector"):
            batch_imgs = batch_imgs.to(device)
            
            # Forward pass: Đã bao gồm F.normalize(p=2) bên trong backbone
            features = model(batch_imgs).cpu().numpy().astype('float32')
            embeddings.append(features)

    # Gộp thành ma trận numpy [N, 256]
    embeddings = np.vstack(embeddings)

    # 5. Khởi tạo FAISS Index
    dimension = 256
    # Dùng IndexFlatIP (Inner Product) cho ArcFace Cosine Similarity
    index = faiss.IndexFlatIP(dimension) 
    
    print("[INFO] Đang đẩy vector vào FAISS Database...")
    index.add(embeddings)

    # 6. Lưu file
    os.makedirs(os.path.dirname(settings.INDEX_PATH), exist_ok=True)
    faiss.write_index(index, settings.INDEX_PATH)

    print(f"[SUCCESS] Hoàn tất! Đã lưu {index.ntotal} vector tại {settings.INDEX_PATH}")

if __name__ == "__main__":
    build_index()