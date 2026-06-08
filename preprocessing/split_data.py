import os
import pandas as pd
import numpy as np

def create_scientific_split(csv_path, save_dir, test_size=0.2, random_state=42):
    print("[*] Đang đọc dữ liệu gốc...")
    df = pd.read_csv(csv_path)
    
    # Lấy danh sách tất cả các nhóm sản phẩm (Mỗi label_group là 1 mặt hàng)
    unique_groups = df['label_group'].unique()
    np.random.seed(random_state)
    np.random.shuffle(unique_groups)
    
    # Cắt mảng nhóm sản phẩm theo tỉ lệ
    split_idx = int(len(unique_groups) * (1 - test_size))
    train_groups = unique_groups[:split_idx]
    val_groups = unique_groups[split_idx:]
    
    # Lọc lại DataFrame dựa trên các nhóm đã chia
    train_df = df[df['label_group'].isin(train_groups)]
    val_df = df[df['label_group'].isin(val_groups)]
    
    print(f"[+] Tổng số ảnh gốc: {len(df)}")
    print(f"    -> Tập Train: {len(train_df)} ảnh ({len(train_groups)} mặt hàng)")
    print(f"    -> Tập Val:   {len(val_df)} ảnh ({len(val_groups)} mặt hàng hoàn toàn mới)")
    
    # Lưu ra 2 file CSV mới
    os.makedirs(save_dir, exist_ok=True)
    train_save_path = os.path.join(save_dir, 'train_split.csv')
    val_save_path = os.path.join(save_dir, 'val_split.csv')
    
    train_df.to_csv(train_save_path, index=False)
    val_df.to_csv(val_save_path, index=False)
    
    print(f"[+] Đã lưu thành công tại: {train_save_path} và {val_save_path}")

if __name__ == "__main__":
    # Đường dẫn trỏ tới thư mục data ở Local
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    DATA_CSV = os.path.join(ROOT_DIR, 'data', 'train.csv')
    SAVE_DIR = os.path.join(ROOT_DIR, 'data')
    
    if not os.path.exists(DATA_CSV):
        print(f"[!] Lỗi: Không tìm thấy file {DATA_CSV}. Vui lòng kiểm tra lại!")
    else:
        create_scientific_split(DATA_CSV, SAVE_DIR)