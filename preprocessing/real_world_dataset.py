# File: preprocessing/real_world_dataset.py

import os
import random
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ShopeeRawTripletDataset(Dataset):
    """
    DataLoader Đã Tối Ưu Tốc Độ CPU (O(1) Lookup)
    """
    def __init__(self, csv_file, img_dir, transform=None):
        print(f"Đang đọc metadata từ: {csv_file}")
        self.data_df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        
        # --- TỐI ƯU HÓA: Đưa data vào Numpy Array / List để truy xuất cực nhanh ---
        self.image_paths = self.data_df['image'].values
        self.labels = self.data_df['label_group'].values
        
        # Tạo Dictionary map: label_group -> danh sách các vị trí (index) của ảnh đó
        self.label_to_indices = self.data_df.groupby('label_group').groups
        # Danh sách các group để random Negative
        self.unique_groups = list(self.label_to_indices.keys())
        
        # NÂNG CẤP 1: Data Augmentation (Đa sản phẩm, chống quá khớp)
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        # 1. Lấy Anchor siêu tốc (Không dùng Pandas)
        anchor_img_name = self.image_paths[index]
        current_group = self.labels[index]
        anchor_path = os.path.join(self.img_dir, anchor_img_name)
        anchor_img = Image.open(anchor_path).convert('RGB')
        
        # 2. Lấy Positive (O(1) random choice từ list index)
        positive_index = random.choice(self.label_to_indices[current_group])
        positive_path = os.path.join(self.img_dir, self.image_paths[positive_index])
        positive_img = Image.open(positive_path).convert('RGB')
        
        # 3. Lấy Negative (O(1) random choice)
        negative_group = random.choice([g for g in self.unique_groups if g != current_group])
        negative_index = random.choice(self.label_to_indices[negative_group])
        negative_path = os.path.join(self.img_dir, self.image_paths[negative_index])
        negative_img = Image.open(negative_path).convert('RGB')
        
        # 4. Transform
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
            
        return anchor_img, positive_img, negative_img, current_group