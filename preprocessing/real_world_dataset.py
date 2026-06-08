# File: preprocessing/real_world_dataset.py

import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder

class ShopeeArcFaceDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """Dataset siêu tốc cho ArcFace: Đọc 1 ảnh 1 nhãn"""
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        
        # Transform mặc định tối ưu
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
            
        # Mã hóa label_group (VD: 12345678) thành index (0, 1, 2... num_classes-1)
        self.label_encoder = LabelEncoder()
        self.df['label_encoded'] = self.label_encoder.fit_transform(self.df['label_group'])
        self.num_classes = len(self.label_encoder.classes_)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image'])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        label = torch.tensor(row['label_encoded'], dtype=torch.long)
        
        return image, label, row['posting_id'], row['label_group']