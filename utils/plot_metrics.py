# File: utils/plot_metrics.py
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_learning_curve():
    csv_path = 'outputs/training_history.csv'
    
    if not os.path.exists(csv_path):
        print(f"[!] Lỗi: Không tìm thấy file {csv_path}.")
        print("Hãy chắc chắn bạn đã tải file này từ Kaggle về thư mục outputs/ ở local.")
        return

    # Đọc dữ liệu
    df = pd.read_csv(csv_path)

    # Khởi tạo biểu đồ chất lượng cao (dpi=300 để in báo cáo không bị mờ)
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

    # Vẽ đường Train Loss (Màu đỏ)
    color = 'tab:red'
    ax1.set_xlabel('Epoch (Vòng lặp huấn luyện)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Train Loss (ArcFace)', color=color, fontsize=12, fontweight='bold')
    ax1.plot(df['epoch'], df['train_loss'], color=color, marker='o', linewidth=2.5, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Vẽ đường Validation mAP@5 (Màu xanh)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Validation mAP@5', color=color, fontsize=12, fontweight='bold')
    ax2.plot(df['epoch'], df['val_map'], color=color, marker='s', linewidth=2.5, label='Val mAP@5')
    ax2.tick_params(axis='y', labelcolor=color)

    # Tiêu đề
    plt.title('Biểu đồ Hội tụ: AI Core (ResNet50 + GeM + ArcFace)', fontsize=15, fontweight='bold', pad=20)
    fig.tight_layout()
    
    # Lưu ảnh ra file
    save_path = 'outputs/training_metrics.png'
    plt.savefig(save_path)
    print(f"[+] Đã xuất biểu đồ chuẩn học thuật tại: {save_path}")
    
    # Hiển thị
    plt.show()

if __name__ == "__main__":
    plot_learning_curve()