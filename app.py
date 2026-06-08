# app.py - Main application file for the Pro E-commerce Search API

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from api.endpoints import router

app = FastAPI(title="Pro E-commerce Search API")

# 1. Cấp quyền CORS để Vue.js (Frontend) có thể gọi API mà không bị block
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong thực tế sẽ điền domain cụ thể, ở Local thì để "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Phục vụ file tĩnh: Cho phép Frontend hiển thị ảnh trực tiếp từ thư mục 'images'
app.mount("/images", StaticFiles(directory="images"), name="images")

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)