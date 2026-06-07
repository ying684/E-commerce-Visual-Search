import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import numpy as np

class FashionMNISTTriplet(Dataset):
    def __init__(self, root='./data', train=True):
        # Biến đổi ảnh để phù hợp với backbone của CNN (224x224, 3 kênh màu RGB)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3), # Chuyển 1 kênh sang 3 kênh
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], # Chuẩn hóa theo ImageNet
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # Tải Fashion-MNIST
        self.dataset = datasets.FashionMNIST(root=root, train=train, download=True, transform=self.transform)
        self.labels = self.dataset.targets.numpy()
        
        # Gom index của các ảnh theo từng class (từ 0 đến 9)
        # Giúp việc tìm ảnh Positive (cùng class) và Negative (khác class) diễn ra trong O(1)
        self.class_indices = {i: np.where(self.labels == i)[0] for i in range(10)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # 1. Lấy Anchor
        anchor_img, label = self.dataset[index]
        
        # 2. Lấy Positive (cùng class, nhưng khác index với Anchor)
        positive_index = index
        while positive_index == index:
            positive_index = np.random.choice(self.class_indices[label])
        positive_img, _ = self.dataset[positive_index]
        
        # 3. Lấy Negative (khác class với Anchor)
        negative_label = np.random.choice([i for i in range(10) if i != label])
        negative_index = np.random.choice(self.class_indices[negative_label])
        negative_img, _ = self.dataset[negative_index]
        
        return anchor_img, positive_img, negative_img, label