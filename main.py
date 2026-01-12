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

def sort_contours_grid(bounding_boxes, tolerance=100):
    """
    Sorts boxes by Row (Y), then by Column (X).
    'tolerance' is the pixel height to consider items as being in the 'same row'.
    """
    # Sort primarily by Y (Row), roughly
    # We bin the Y values: y // tolerance. 
    # This groups items that are within 100px vertically of each other.
    return sorted(bounding_boxes, key=lambda b: (b[1] // tolerance, b[0]))

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
        img = cv2.cvtColor(np.array(page_image_pil), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

        # --- STEP 1: ERASE CROSSES ---
        contours_all, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a working mask without crosses
        clean_mask = thresh.copy()
        
        for cnt in contours_all:
            x, y, w, h = cv2.boundingRect(cnt)
            # Detect Crosses: Small box (<100px)
            if w < 100 and h < 100:
                # DRAW BLACK (0) over the cross to erase it from the mask
                cv2.drawContours(clean_mask, [cnt], -1, 0, -1)

        # --- STEP 2: CLUSTER CONTENT ---
        # We dilate the clean mask to connect Stamp+Address
        # Kernel: (20, 50) -> Connects small horizontal gaps (20), LARGE vertical gaps (50)
        # This ensures the Stamp connects to the Address below it, 
        # but is unlikely to jump the wide gap to the next column.
        kernel = np.ones((50, 20), np.uint8) 
        dilated = cv2.dilate(clean_mask, kernel, iterations=2)
        
        # --- STEP 3: EXTRACT BLOCKS ---
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        
        # --- STEP 4: SMART SORTING (GRID AWARE) ---
        # Sorts: Top-Left -> Top-Right -> Bottom-Left...
        bounding_boxes = sort_contours_grid(bounding_boxes, tolerance=100)

        for (x, y, w, h) in bounding_boxes:
            # Filter:
            # Must be reasonably big (ignore noise/dust)
            if w < 150 or h < 100: continue
            
            # Crop Logic
            padding = 20
            h_img, w_img, _ = img.shape
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w_img, x + w + padding)
            y2 = min(h_img, y + h + padding)
            
            stamp_crop = img[y1:y2, x1:x2]
            
            # Convert
            stamp_crop_rgb = cv2.cvtColor(stamp_crop, cv2.COLOR_BGR2RGB)
            pil_image = ImageReader(Image.fromarray(stamp_crop_rgb))

            # --- PLACEMENT ---
            img_h_crop, img_w_crop, _ = stamp_crop.shape
            
            # Fit on A6 (90% width/height)
            target_w = a6_width * 0.9
            target_h = a6_height * 0.9
            
            scale = min(1.0, target_w / img_w_crop, target_h / img_h_crop)
            draw_w = img_w_crop * scale
            draw_h = img_h_crop * scale
            
            # Center on Page
            x_pos = (a6_width - draw_w) / 2
            y_pos = (a6_height - draw_h) / 2
            
            c.drawImage(pil_image, x_pos, y_pos, width=draw_w, height=draw_h)
            c.showPage()
            total_stamps += 1

    c.save()
    output_buffer.seek(0)
    
    filename = f"Stamps_Count_{total_stamps}.pdf"
    
    return StreamingResponse(
        output_buffer, 
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )
