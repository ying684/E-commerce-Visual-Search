import os
import pandas as pd
from PIL import Image
from preprocessing.real_world_dataset import CrossDomainTripletDataset

def create_mock_data():
    """Tạo thư mục dummy_data, vài tấm ảnh trống và file CSV để test luồng logic"""
    os.makedirs('dummy_data', exist_ok=True)
    
    # Tạo vài ảnh RGB trống đại diện cho ảnh thật
    img_names = ['q_shirt1.jpg', 's_shirt1.jpg', 'q_pants1.jpg', 's_pants1.jpg']
    for name in img_names:
        Image.new('RGB', (100, 100), color=(73, 109, 137)).save(f'dummy_data/{name}')
        
    # Tạo file CSV mapping
    data = {
        'item_id': ['item_001', 'item_002'],
        'query_path': ['q_shirt1.jpg', 'q_pants1.jpg'], # Ảnh khách chụp
        'shop_path': ['s_shirt1.jpg', 's_pants1.jpg']   # Ảnh shop
    }
    df = pd.DataFrame(data)
    df.to_csv('dummy_data/mapping.csv', index=False)
    print("Đã tạo Mock Data thành công!")

def test_module():
    create_mock_data()
    
    print("\nKhởi tạo CrossDomain Dataset...")
    dataset = CrossDomainTripletDataset(
        csv_file='dummy_data/mapping.csv', 
        img_dir='dummy_data/'
    )
    
    anchor, positive, negative, item_id = dataset[0]
    
    print(f"Tổng số cặp dữ liệu: {len(dataset)}")
    print(f"Item ID của mẫu test: {item_id}")
    print(f"Kích thước Anchor: {anchor.shape} (Kỳ vọng [3, 224, 224])")
    print(f"Kích thước Positive: {positive.shape}")
    print(f"Kích thước Negative: {negative.shape}")
    print("\n[THÀNH CÔNG] Module xử lý ảnh thực tế đã sẵn sàng để ráp vào Training Loop!")

if __name__ == "__main__":
    test_module()