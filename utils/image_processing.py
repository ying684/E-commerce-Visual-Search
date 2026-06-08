# utils/image_processing.py

from PIL import Image

def process_image(image_path):
    """Xử lý ảnh (lót nền trắng cho PNG/WebP trong suốt)"""
    img = Image.open(image_path)
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        alpha = img.convert('RGBA').split()[-1]
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=alpha)
        return bg
    return img.convert('RGB')