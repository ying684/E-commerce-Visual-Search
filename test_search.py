from inference import VisualSearchEngine
import json

engine = VisualSearchEngine(
    model_path='outputs/best_cbir_model.pth',
    csv_path='/kaggle/input/competitions/shopee-product-matching/train.csv',
    img_dir='/kaggle/input/competitions/shopee-product-matching/train_images/'
)

# Lấy thử 1 ảnh làm Query
query_img = '/kaggle/input/competitions/shopee-product-matching/train_images/0000a68812bc7e98c42888dfb1c07da0.jpg'

# Chạy tìm kiếm
output_data = engine.search(query_image_path=query_img, top_k=5)

# In kết quả dưới dạng JSON đẹp mắt
print(json.dumps(output_data, indent=4))