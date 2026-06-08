import torch
import faiss
import pandas as pd
import numpy as np
from torchvision import transforms
from models.backbone import CBIRBackbone
from core.config import settings
from utils.image_processing import process_image

class VisualSearchEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CBIRBackbone(model_name='resnet50', embedding_dim=256)
        self.model.load_state_dict(torch.load(settings.MODEL_PATH, map_location=self.device))
        self.model.to(self.device).eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.df = pd.read_csv(settings.CSV_PATH)
        self.index = faiss.read_index(settings.INDEX_PATH)

    def search(self, image_path, top_k):
        img = process_image(image_path)
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            vector = self.model(tensor).cpu().numpy().astype('float32')
            
        distances, indices = self.index.search(vector, top_k)
        
        results = []
        for i in range(top_k):
            idx = indices[0][i]
            results.append({
                "rank": i + 1,
                "posting_id": self.df['posting_id'].iloc[idx],
                "distance": float(distances[0][i])
            })
        return results