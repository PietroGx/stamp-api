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

def get_projection_zones(binary_img, axis=1, gap_tolerance=30, min_size=100):
    """
    Generic function to find 'zones' of content based on projection.
    axis=1: Horizontal Projection (Finding Rows)
    axis=0: Vertical Projection (Finding Columns)
    """
    # 1. Sum pixels along axis (Count non-zero/white pixels)
    projection = np.sum(binary_img, axis=axis)
    
    # 2. Threshold: What counts as "empty"?
    # Allow 1% noise
    max_val = binary_img.shape[axis] * 255 if axis == 0 else binary_img.shape[1] * 255
    empty_thresh = max_val * 0.01

    has_content = projection > empty_thresh
    indices = np.arange(len(has_content))
    content_indices = indices[has_content]
    
    if len(content_indices) == 0:
        return []

    # 3. Group into zones
    zones = []
    start = content_indices[0]
    last = content_indices[0]
    
    for i in content_indices[1:]:
        if i - last > gap_tolerance:
            # Large gap found -> Close current zone
            if (last - start) > min_size: # Filter tiny noise blocks
                zones.append((start, last))
            start = i
        last = i
    
    # Append final zone
    if (last - start) > min_size:
        zones.append((start, last))
        
    return zones

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
        
        # Binarize (White Text on Black Background)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        # --- STEP 1: DETECT CROSSES (For Hybrid Logic) ---
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cross_y_positions = []
        
        # Create a "Clean" mask (No crosses) for projection analysis
        clean_thresh = thresh.copy()

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Identify Crosses (Small ~50px boxes)
            if 10 < w < 80 and 10 < h < 80:
                cross_y_positions.append(y + h//2)
                # Erase cross from analysis mask
                cv2.drawContours(clean_thresh, [cnt], -1, 0, -1)

        cross_y_positions.sort()
        
        # Filter close Y duplicates
        unique_cross_rows = []
        if cross_y_positions:
            last = -100
            for y in cross_y_positions:
                if abs(y - last) > 50:
                    unique_cross_rows.append(y)
                    last = y

        # --- STEP 2: ROW SLICING (Hybrid) ---
        row_slices = [] # List of (y1, y2)
        
        # LOGIC: If we have > 2 cross rows (Top, Middle, Bottom...), use Crosses.
        # Otherwise (only Top/Bottom or None), use White Gap Projection.
        if len(unique_cross_rows) > 2:
            # Use Crosses as dividers
            # We skip the first and last "zones" if they are outside the crosses
            for i in range(len(unique_cross_rows) - 1):
                y1 = unique_cross_rows[i]
                y2 = unique_cross_rows[i+1]
                # Filter small slices
                if (y2 - y1) > 150:
                    # Shrink slightly to avoid the cross line itself
                    row_slices.append((y1 + 20, y2 - 20))
        else:
            # Use Projection (White Gaps)
            # axis=1 for rows. gap_tolerance=40 (vertical gap size)
            row_slices = get_projection_zones(clean_thresh, axis=1, gap_tolerance=40, min_size=150)

        # --- STEP 3: COLUMN SLICING (Vertical Projection) ---
        for (r_y1, r_y2) in row_slices:
            # Extract the Row Image
            row_img = img[r_y1:r_y2, :]
            row_thresh = clean_thresh[r_y1:r_y2, :]
            
            # Look for columns inside this row
            # axis=0 for columns. gap_tolerance=50 (horizontal gap size)
            col_slices = get_projection_zones(row_thresh, axis=0, gap_tolerance=50, min_size=150)
            
            for (c_x1, c_x2) in col_slices:
                # Padding
                pad = 10
                final_y1 = max(0, r_y1 - pad)
                final_y2 = min(img.shape[0], r_y2 + pad)
                final_x1 = max(0, c_x1 - pad)
                final_x2 = min(img.shape[1], c_x2 + pad)
                
                stamp_crop = img[final_y1:final_y2, final_x1:final_x2]
                
                # --- PDF PLACEMENT ---
                stamp_crop_rgb = cv2.cvtColor(stamp_crop, cv2.COLOR_BGR2RGB)
                pil_image = ImageReader(Image.fromarray(stamp_crop_rgb))

                h_crop, w_crop, _ = stamp_crop.shape
                
                # Fit to A6
                target_w = a6_width * 0.95
                target_h = a6_height * 0.95
                
                scale = min(1.0, target_w / w_crop, target_h / h_crop)
                draw_w = w_crop * scale
                draw_h = h_crop * scale
                
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
