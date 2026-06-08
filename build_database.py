import os
import torch
import faiss
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image, ImageFile

# [QUAN TRỌNG] Bỏ qua lỗi nếu có file ảnh bị hỏng/tải thiếu dung lượng trong 32k ảnh
ImageFile.LOAD_TRUNCATED_IMAGES = True

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
        img_name = self.df['image'].iloc[idx]
        img_path = os.path.join(self.image_dir, img_name)
        img = process_image(img_path)
        
        if self.transform:
            img = self.transform(img)
            
        return img

def build_index():
    # KIỂM TRA CARD ĐỒ HỌA
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[🔥 TỐC ĐỘ BÀN THỜ] Đã nhận diện GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[⚠️ CẢNH BÁO] Vẫn đang chạy bằng CPU! Hãy kiểm tra lại bản cài PyTorch.")

    model = CBIRBackbone(model_name='resnet50', embedding_dim=256)
    model.load_state_dict(torch.load(settings.MODEL_PATH, map_location=device))
    model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print(f"[INFO] Đọc metadata từ: {settings.CSV_PATH}")
    df = pd.read_csv(settings.CSV_PATH)
    
    dataset = CBIRDataset(df, settings.IMG_DIR, transform=transform)
    
    # [TỐI ƯU CHO GTX 1650]: batch_size=32 để chống tràn VRAM (4GB), num_workers=0 để chống lỗi trên Windows
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    embeddings = []

    with torch.no_grad():
        for batch_imgs in tqdm(dataloader, desc="Đang trích xuất vector"):
            batch_imgs = batch_imgs.to(device)
            features = model(batch_imgs).cpu().numpy().astype('float32')
            embeddings.append(features)

    embeddings = np.vstack(embeddings)

    dimension = 256
    index = faiss.IndexFlatIP(dimension) 
    
    print("[INFO] Đang đẩy vector vào FAISS Database...")
    index.add(embeddings)

    os.makedirs(os.path.dirname(settings.INDEX_PATH), exist_ok=True)
    faiss.write_index(index, settings.INDEX_PATH)

    print(f"[SUCCESS] Hoàn tất! Đã lưu {index.ntotal} vector tại {settings.INDEX_PATH}")

if __name__ == "__main__":
    build_index()