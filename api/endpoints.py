import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from core.config import settings
from services.search_engine import VisualSearchEngine

router = APIRouter()
engine = VisualSearchEngine()

@router.post("/search")
async def search(file: UploadFile = File(...), top_k: int = 5):
    if file.content_type not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(400, "Định dạng không hỗ trợ.")
        
    temp_path = os.path.join(settings.TEMP_DIR, f"{uuid.uuid4().hex}_{file.filename}")
    
    try:
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())
        return {"results": engine.search(temp_path, top_k)}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)