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

    # 2. Convert
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=300)
    except:
        raise HTTPException(status_code=500, detail="PDF conversion failed")

    output_buffer = io.BytesIO()
    c = canvas.Canvas(output_buffer, pagesize=landscape(A6))
    a6_width, a6_height = landscape(A6)
    
    total_stamps = 0

    for page_image_pil in pages:
        # Convert to standard OpenCV format
        img = cv2.cvtColor(np.array(page_image_pil), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # --- STEP 1: REMOVE CROSSES ---
        # Find all contours first
        contours_all, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a "clean" mask where we will erase the crosses
        clean_mask = thresh.copy()
        
        for cnt in contours_all:
            x, y, w, h = cv2.boundingRect(cnt)
            # A cross is usually small (e.g., < 80px). A stamp/address is wide (> 200px).
            # If we see a small box, we assume it's a cross/separator and ERASE it.
            if w < 100 and h < 100:
                cv2.drawContours(clean_mask, [cnt], -1, (0, 0, 0), -1)  # Draw black over it

        # --- STEP 2: MERGE CONTENT ---
        # Now we dilate the "clean" mask (which has no crosses).
        # We use a kernel that connects vertical lines (Address) + horizontal (Stamp text)
        # (15, 100) = Connects lines 15px apart vertically, and words 100px apart horizontally.
        kernel = np.ones((15, 100), np.uint8) 
        dilated = cv2.dilate(clean_mask, kernel, iterations=2)
        
        # --- STEP 3: EXTRACT FINAL BLOCKS ---
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        bounding_boxes.sort(key=lambda x: x[1]) # Sort Top-to-Bottom

        for (x, y, w, h) in bounding_boxes:
            # Final Safety Filter:
            # Only keep things that are big enough to be a Stamp+Address
            if w < 200 or h < 100: continue
            
            # Crop Logic
            padding = 10
            h_img, w_img, _ = img.shape
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w_img, x + w + padding)
            y2 = min(h_img, y + h + padding)
            
            stamp_crop = img[y1:y2, x1:x2]
            
            # Convert for PDF
            stamp_crop_rgb = cv2.cvtColor(stamp_crop, cv2.COLOR_BGR2RGB)
            pil_image = ImageReader(Image.fromarray(stamp_crop_rgb))

            # --- PLACEMENT ---
            # Center-Right positioning
            img_w, img_h = x2-x1, y2-y1
            target_w = a6_width * 0.7  # Use 70% of width
            target_h = a6_height * 0.9 # Use 90% of height
            
            scale = min(1.0, target_w / img_w, target_h / img_h)
            
            draw_w, draw_h = img_w * scale, img_h * scale
            x_pos = a6_width - draw_w - 20
            y_pos = a6_height - draw_h - 20
            
            c.drawImage(pil_image, x_pos, y_pos, width=draw_w, height=draw_h)
            c.showPage()
            total_stamps += 1

    c.save()
    output_buffer.seek(0)
    
    # Filename Hack for Zapier
    filename = f"Letters_printout_({total_stamps}).pdf"
    
    return StreamingResponse(
        output_buffer, 
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )
