# E-commerce Visual Search API

Hệ thống tìm kiếm sản phẩm bằng hình ảnh (CBIR) sử dụng Deep Learning và Vector Database.

## 🚀 Công nghệ sử dụng
- **Backend:** Python, FastAPI, PyTorch (ResNet50, ArcFace).
- **Search Engine:** FAISS (để tìm kiếm vector tốc độ cao).
- **Frontend:** Vue.js.

## 🛠️ Hướng dẫn chạy dự án

### 1. Backend
- Mở Terminal tại thư mục gốc của dự án.
- Tạo môi trường ảo (khuyên dùng):
  ```bash
  python -m venv venv
  source venv/bin/activate  # Trên Windows: venv\Scripts\activate
Cài đặt thư viện:

Bash
pip install -r requirements.txt
Chạy server:

Bash
python app.py
### 2. Frontend
Mở một Terminal mới, chuyển vào thư mục frontend:

Bash
cd frontend
Cài đặt node modules:

Bash
npm install
Chạy ứng dụng:

Bash
npm run dev
### 📁 Lưu ý
Đảm bảo các file model (.pth) và file index (.index) đã có trong thư mục outputs/.

Nếu chưa có file .index, hãy chạy lệnh python build_database.py để hệ thống tự quét và tạo dữ liệu tìm kiếm.