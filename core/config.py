import os

class Settings:
    MODEL_PATH = "outputs/best_cbir_model.pth"
    CSV_PATH = "data/train.csv"
    INDEX_PATH = "outputs/faiss_database.index"
    IMG_DIR = "images/"
    TEMP_DIR = "temp/"
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
    ALLOWED_EXTENSIONS = {"image/jpeg", "image/png", "image/webp"}

settings = Settings()
os.makedirs(settings.TEMP_DIR, exist_ok=True)