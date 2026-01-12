import cv2
import numpy as np
import io
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from pdf2image import convert_from_bytes
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A6, landscape
from reportlab.lib.utils import ImageReader
from PIL import Image

app = FastAPI()

# --- CONFIGURATION ---
MIN_GAP_HEIGHT = 20
MAX_GAP_HEIGHT = 60
GAP_THRESHOLD = 0.04
MIN_STAMP_HEIGHT = 400

def crop_to_bottom_content(image):
    """
    Scans the image to find the lowest non-white pixel and crops
    the image at that line.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(thresh)
    
    if coords is None:
        return image 
        
    x, y, w, h = cv2.boundingRect(coords)
    bottom_y = y + h
    cut_line = min(bottom_y + 20, image.shape[0])
    
    return image[0:cut_line, :]

def find_gap_centers(projection, threshold_value, min_gap_width=10, max_gap_width=1000):
    norm = projection / (np.max(projection) + 1e-5)
    is_gap = norm < threshold_value
    gap_centers = []
    current_gap_start = -1
    
    for i, val in enumerate(is_gap):
        if val:
            if current_gap_start == -1: current_gap_start = i
        else:
            if current_gap_start != -1:
                gap_width = i - current_gap_start
                if min_gap_width < gap_width < max_gap_width:
                    center = current_gap_start + gap_width // 2
                    gap_centers.append(center)
                current_gap_start = -1
                
    if current_gap_start != -1:
        gap_width = len(projection) - current_gap_start
        if min_gap_width < gap_width < max_gap_width:
            gap_centers.append(current_gap_start + gap_width // 2)
            
    return gap_centers

def merge_close_cuts(cuts, total_length, min_dist):
    if not cuts: return []
    merged_cuts = []
    last_accepted = 0
    for cut in cuts:
        segment_size = cut - last_accepted
        if segment_size < min_dist:
            continue
        else:
            merged_cuts.append(cut)
            last_accepted = cut
    return merged_cuts

@app.post("/process-pdf/")
async def process_pdf(file: UploadFile = File(...)):
    # Read uploaded file
    file_bytes = await file.read()
    
    try:
        # Convert PDF bytes to images directly in memory
        pages = convert_from_bytes(file_bytes, dpi=300)
    except Exception as e:
        return {"error": f"Error reading PDF: {str(e)}"}

    # Prepare Output PDF in Memory
    output_buffer = io.BytesIO()
    c = canvas.Canvas(output_buffer, pagesize=landscape(A6))
    a6_w, a6_h = landscape(A6)
    
    total_stamps = 0

    for page_image_pil in pages:
        # Convert PIL to OpenCV format
        img_full = cv2.cvtColor(np.array(page_image_pil), cv2.COLOR_RGB2BGR)
        
        # 1. PRE-CROP
        img = crop_to_bottom_content(img_full)
        
        h_img, w_img, _ = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        row_proj = np.sum(thresh, axis=1)
        col_proj = np.sum(thresh, axis=0)
        
        # 2. FIND CUTS
        raw_y_cuts = find_gap_centers(row_proj, GAP_THRESHOLD, 
                                      min_gap_width=MIN_GAP_HEIGHT, 
                                      max_gap_width=MAX_GAP_HEIGHT)
        raw_x_cuts = find_gap_centers(col_proj, GAP_THRESHOLD, 
                                      min_gap_width=20, 
                                      max_gap_width=2000)
        
        # 3. MERGE
        final_y_cuts = merge_close_cuts(raw_y_cuts, h_img, MIN_STAMP_HEIGHT)
        final_x_cuts = merge_close_cuts(raw_x_cuts, w_img, 200)

        # 4. SLICE & LAYOUT
        all_x = [0] + final_x_cuts + [w_img]
        all_y = [0] + final_y_cuts + [h_img]
        
        for i in range(len(all_y) - 1):
            for j in range(len(all_x) - 1):
                y1, y2 = all_y[i], all_y[i+1]
                x1, x2 = all_x[j], all_x[j+1]
                
                if (y2 - y1) < MIN_STAMP_HEIGHT: continue
                if (x2 - x1) < 200: continue
                
                stamp = img[y1+5:y2-5, x1+5:x2-5]
                
                if np.mean(stamp) > 250: continue 

                stamp_rgb = cv2.cvtColor(stamp, cv2.COLOR_BGR2RGB)
                pil_stamp = ImageReader(Image.fromarray(stamp_rgb))
                
                crop_h, crop_w, _ = stamp.shape
                target_w = a6_w * 0.5
                target_h = a6_h * 0.5
                scale = min(1.0, target_w / crop_w, target_h / crop_h)
                
                final_w = crop_w * scale
                final_h = crop_h * scale
                
                margin = 20
                x_pos = a6_w - final_w - margin
                y_pos = a6_h - final_h - margin
                
                c.drawImage(pil_stamp, x_pos, y_pos, width=final_w, height=final_h)
                c.showPage()
                total_stamps += 1

    c.save()
    output_buffer.seek(0)
    
    return StreamingResponse(
        output_buffer, 
        media_type="application/pdf", 
        headers={"Content-Disposition": f"attachment; filename=printout_stamps_{total_stamps}.pdf"}
    )
