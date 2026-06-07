import os
import random
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CrossDomainTripletDataset(Dataset):
    """
    Dataset thực tiễn xử lý ảnh E-commerce.
    Yêu cầu một file DataFrame (CSV) chứa 3 cột:
    - query_path: Đường dẫn tới ảnh người dùng chụp (Anchor)
    - shop_path: Đường dẫn tới ảnh của cửa hàng (Positive)
    - item_id: ID duy nhất của sản phẩm
    """
    def __init__(self, csv_file, img_dir, transform=None):
        self.data_df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        
        # Danh sách toàn bộ item_id độc nhất để phục vụ việc chọn Negative
        self.unique_items = self.data_df['item_id'].unique()
        
        # Nhóm dữ liệu theo item_id để truy xuất siêu tốc (O(1))
        self.item_groups = self.data_df.groupby('item_id')
        
        # Transform chuẩn cho CNN (ResNet/MobileNet)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        # 1. Lấy thông tin dòng hiện tại
        row = self.data_df.iloc[index]
        current_item_id = row['item_id']
        
        # Lấy Anchor (Ảnh khách chụp)
        anchor_path = os.path.join(self.img_dir, row['query_path'])
        anchor_img = Image.open(anchor_path).convert('RGB')
        
        # 2. Lấy Positive (Ảnh shop chụp của cùng 1 item_id)
        # Thực tế 1 item có thể có nhiều ảnh shop, ta chọn ngẫu nhiên 1 tấm
        positive_row = self.item_groups.get_group(current_item_id).sample(1).iloc[0]
        positive_path = os.path.join(self.img_dir, positive_row['shop_path'])
        positive_img = Image.open(positive_path).convert('RGB')
        
        # 3. Lấy Negative (Ảnh của một item_id hoàn toàn khác)
        negative_item_id = random.choice([i for i in self.unique_items if i != current_item_id])
        negative_row = self.item_groups.get_group(negative_item_id).sample(1).iloc[0]
        
        # Chọn ngẫu nhiên lấy ảnh shop hay ảnh query của item khác làm Negative
        neg_col = random.choice(['query_path', 'shop_path'])
        negative_path = os.path.join(self.img_dir, negative_row[neg_col])
        negative_img = Image.open(negative_path).convert('RGB')
        
        # 4. Áp dụng Transform
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
            
        return anchor_img, positive_img, negative_img, current_item_id