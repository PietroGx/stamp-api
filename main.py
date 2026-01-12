from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import requests
import cv2
import numpy as np
from pdf2image import convert_from_bytes
from reportlab.lib.pagesizes import A6, landscape
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from PIL import Image
import io

app = FastAPI()

class PdfRequest(BaseModel):
    pdfstampurl: str

@app.post("/process-stamps")
def process_stamps(request: PdfRequest):
    # 1. Download
    try:
        response = requests.get(request.pdfstampurl)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Could not download PDF")
        pdf_bytes = response.content
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 2. Convert to Images
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=300)
    except:
        raise HTTPException(status_code=500, detail="PDF conversion failed")

    # 3. Process
    output_buffer = io.BytesIO()
    c = canvas.Canvas(output_buffer, pagesize=landscape(A6))
    a6_width, a6_height = landscape(A6)
    
    # Track the count
    total_stamps_found = 0

    for page_image_pil in pages:
        img = cv2.cvtColor(np.array(page_image_pil), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((30, 30), np.uint8) 
        dilated = cv2.dilate(thresh, kernel, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        bounding_boxes.sort(key=lambda x: x[1])

        for (x, y, w, h) in bounding_boxes:
            if w < 250 or h < 100: continue
            
            padding = 30
            h_img, w_img, _ = img.shape
            x1 = max(0, x - padding); y1 = max(0, y - padding)
            x2 = min(w_img, x + w + padding); y2 = min(h_img, y + h + padding)
            
            stamp_crop = img[y1:y2, x1:x2]
            stamp_crop_rgb = cv2.cvtColor(stamp_crop, cv2.COLOR_BGR2RGB)
            pil_image = ImageReader(Image.fromarray(stamp_crop_rgb))

            img_w, img_h = x2-x1, y2-y1
            target_w, target_h = a6_width * 0.5, a6_height * 0.5
            scale = min(1.0, target_w / img_w, target_h / img_h)
            
            draw_w, draw_h = img_w * scale, img_h * scale
            x_pos = a6_width - draw_w - 20
            y_pos = a6_height - draw_h - 20
            
            c.drawImage(pil_image, x_pos, y_pos, width=draw_w, height=draw_h)
            c.showPage()
            
            # Increment Counter
            total_stamps_found += 1

    c.save()
    output_buffer.seek(0)
    
    # 4. Return File + Header
    return StreamingResponse(
        output_buffer, 
        media_type="application/pdf",
        headers={
            "Content-Disposition": "attachment; filename=stamps.pdf",
            "X-Page-Count": str(total_stamps_found)  # <--- HERE IS THE NEW DATA
        }
    )
