# E-commerce Visual Search API

Hệ thống tìm kiếm sản phẩm bằng hình ảnh (**CBIR — Content-Based Image Retrieval**) cho thương mại điện tử. Người dùng tải lên một bức ảnh sản phẩm, hệ thống sẽ trích xuất đặc trưng (embedding) bằng mạng deep learning và tìm các sản phẩm tương tự nhất trong cơ sở dữ liệu vector bằng FAISS.

## 🚀 Công nghệ sử dụng

| Thành phần | Công nghệ |
|---|---|
| **Backend API** | Python, FastAPI, Uvicorn |
| **Mô hình trích xuất đặc trưng** | PyTorch, ResNet50 (backbone) + GeM Pooling, ArcFace Loss |
| **Tìm kiếm vector** | FAISS (`IndexFlatIP` — Inner Product trên embedding đã chuẩn hoá L2) |
| **Frontend** | Vue.js 3, Vite, Tailwind CSS |

## 📋 Yêu cầu hệ thống

- **Python 3.11**
- **Node.js 18+** (khuyến nghị 20+) và npm — để chạy frontend
- **(Tuỳ chọn) GPU NVIDIA + driver CUDA** — giúp train và inference mô hình PyTorch nhanh hơn đáng kể. Nếu không có GPU, dự án vẫn chạy được trên CPU (FAISS dùng `faiss-cpu`)
- **Tài khoản Kaggle** — cần thiết để tải bộ dữ liệu ảnh sản phẩm (xem mục [Chuẩn bị dữ liệu](#-chuẩn-bị-dữ-liệu))

## 🛠️ Hướng dẫn cài đặt Backend

Mở Terminal tại thư mục gốc của dự án. Bạn có thể chọn **một trong hai cách** dưới đây để tạo môi trường Python cô lập.

### Cách 1: Dùng `venv` (tích hợp sẵn trong Python)

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Cách 2 (★ Khuyến nghị): Dùng Miniconda / Anaconda

Miniconda được khuyến nghị vì:
- Quản lý tốt các thư viện có phụ thuộc hệ thống phức tạp như **PyTorch + CUDA** (tự động cài đúng phiên bản CUDA Toolkit tương thích, tránh xung đột với driver hệ thống)
- Cú pháp tạo/kích hoạt môi trường **giống hệt nhau trên cả Windows, macOS và Linux** sau khi đã cài đặt
- Dễ dàng cài lại / xoá môi trường khi gặp lỗi mà không ảnh hưởng tới Python hệ thống

**Bước 1 — Cài đặt Miniconda** (nếu chưa có): tải bộ cài phù hợp với hệ điều hành của bạn (Windows / macOS / Linux) tại trang chủ Miniconda:
👉 https://docs.conda.io/projects/miniconda/en/latest/

Sau khi tải, chạy file cài đặt và làm theo hướng dẫn mặc định.

**Bước 2 — Tạo và kích hoạt môi trường** (chạy trong Anaconda Prompt trên Windows, hoặc Terminal trên macOS/Linux — lệnh giống nhau cho cả 3 OS):
```bash
conda create -n cbir-search python=3.11
conda activate cbir-search
```

### Cài đặt thư viện

Sau khi đã kích hoạt môi trường ảo (bằng venv hoặc conda):
```bash
pip install -r requirements.txt
```

> 💡 **Lưu ý:** Nếu bạn có GPU NVIDIA và muốn PyTorch tận dụng CUDA, hãy cài `torch`/`torchvision` theo đúng phiên bản CUDA của máy trước khi chạy lệnh trên (xem hướng dẫn tại trang chủ PyTorch — https://pytorch.org/get-started/locally/), sau đó chạy lại `pip install -r requirements.txt` để cài các thư viện còn lại.

## 📦 Chuẩn bị dữ liệu

Dự án cần 2 thành phần dữ liệu **không có sẵn trong repo Git** (do dung lượng lớn, đã được khai báo trong `.gitignore`):

### 1. Thư mục `images/` — bộ ảnh sản phẩm

Backend cần thư mục `images/` ở thư mục gốc dự án để phục vụ ảnh sản phẩm (tham chiếu trong [core/config.py](core/config.py) và mount tại [app.py](app.py)). Bộ dữ liệu này khớp với cuộc thi Kaggle **Shopee - Product Matching**:
👉 https://www.kaggle.com/competitions/shopee-product-matching/data

Cách tải qua Kaggle API:
1. Đăng nhập Kaggle, vào trang luật của cuộc thi (https://www.kaggle.com/competitions/shopee-product-matching/rules) và bấm **"I Understand and Accept"** để tham gia (bắt buộc, nếu không API sẽ báo lỗi `403 Forbidden` khi tải)
2. Tạo API token tại trang **Account** của Kaggle, tải file `kaggle.json` và đặt vào:
   - Linux/macOS: `~/.kaggle/kaggle.json`
   - Windows: `C:\Users\<tên-user>\.kaggle\kaggle.json`
3. Cài CLI và tải dữ liệu:
   ```bash
   pip install kaggle
   kaggle competitions download -c shopee-product-matching -p data/kaggle_download
   ```
4. Giải nén file `shopee-product-matching.zip` vừa tải, sau đó **đổi tên thư mục `train_images/` thành `images/`** và đặt ở thư mục gốc dự án (ngang hàng với `app.py`). Tên file ảnh trong thư mục này phải khớp với cột `image` trong [data/train.csv](data/train.csv) (ví dụ: `0000a68812bc7e98c42888dfb1c07da0.jpg`)

### 2. Thư mục `outputs/` — model đã huấn luyện và FAISS index

Đảm bảo các file sau đã có trong `outputs/`:
- `best_cbir_model.pth` — trọng số mô hình CBIR đã huấn luyện
- `faiss_database.index` — FAISS index chứa embedding của toàn bộ sản phẩm

Nếu chưa có file `.index` (hoặc muốn build lại từ đầu sau khi có `images/` và model), chạy:
```bash
python build_database.py
```
Script này sẽ quét toàn bộ ảnh trong `images/`, trích xuất embedding bằng mô hình trong `outputs/best_cbir_model.pth`, rồi tạo file `outputs/faiss_database.index`.

## ▶️ Chạy Backend

```bash
python app.py
```
Server sẽ chạy tại `http://127.0.0.1:8000`.

## 🎨 Chạy Frontend

Mở một Terminal mới, chuyển vào thư mục `frontend`:
```bash
cd frontend
npm install
npm run dev
```

## ❓ Xử lý lỗi thường gặp

**1. `ERROR: No matching distribution found for faiss-gpu` khi `pip install -r requirements.txt`**
Gói `faiss-gpu` đã ngừng phát hành trên PyPI (chỉ còn phân phối qua conda). Dự án này dùng `faiss-cpu` trong [requirements.txt](requirements.txt) — chạy thuần CPU vẫn hoạt động đầy đủ tính năng tìm kiếm, chỉ chậm hơn một chút với index rất lớn.

**2. `RuntimeError: Directory 'images' does not exist` khi chạy `python app.py`**
Thư mục `images/` chưa được tạo/tải về. Làm theo hướng dẫn ở mục [Chuẩn bị dữ liệu](#-chuẩn-bị-dữ-liệu) phía trên để tải bộ ảnh từ Kaggle và đặt đúng vị trí.
