# File: utils/logger.py
import csv
import os

class TrainingLogger:
    def __init__(self, save_path='outputs/history.csv'):
        self.save_path = save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Tạo header nếu file chưa tồn tại
        if not os.path.exists(save_path):
            with open(save_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['epoch', 'train_loss', 'val_map', 'learning_rate'])
                
    def log(self, epoch, train_loss, val_map, learning_rate):
        with open(self.save_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, train_loss, val_map, learning_rate])