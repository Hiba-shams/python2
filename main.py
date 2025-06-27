from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from app.ocr_utils import extract_text, validate_license_fields
from app.preprocessing import preprocess_image
import shutil
import os
import uuid

from app.face_utils import ImageQualityInspector  # <-- Updated import!

app = FastAPI()

# Initialize the smart inspector (you can adjust thresholds if needed)
inspector = ImageQualityInspector(debug=False)

# Mount images directory for static access
app.mount("/images", StaticFiles(directory="images"), name="images")

@app.post("/analyze-id")
async def analyze_id(image: UploadFile = File(...)):
    # Save uploaded image
    unique_filename = f"{uuid.uuid4()}.jpg"
    image_dir = "images"
    os.makedirs(image_dir, exist_ok=True)
    image_path = os.path.join(image_dir, unique_filename)

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    # Step 1: Face detection + Blur detection (with smart inspector)
    results = inspector.evaluate_image(image_path)

    face_ok, face_message = results['face_check']
    blurry_ok, blur_message = results['blurry_check']

    if not face_ok:
        return {
            "status": "rejected",
            "message": f"Face check failed: {face_message}",
            "original_image_url": f"/images/{os.path.basename(image_path)}"
        }

    if not blurry_ok:
        return {
            "status": "rejected",
            "message": f"Image quality issue: {blur_message}",
            "original_image_url": f"/images/{os.path.basename(image_path)}"
        }

    # Step 2: OCR + Validation
    try:
        preprocessed_path = preprocess_image(image_path)
        extracted_text = extract_text(preprocessed_path)
        is_valid, fields = validate_license_fields(extracted_text)

        return {
            "status": "accepted" if is_valid else "rejected",
            "message": "Valid driver’s license" if is_valid else "The uploaded image does not appear to be a valid license.",
            "fields": fields,
            "original_image_url": f"/images/{os.path.basename(image_path)}"
        }
    except Exception as e:
        return {
            "status": "rejected",
            "message": f"Processing failed: {str(e)}",
            "original_image_url": f"/images/{os.path.basename(image_path)}"
        }
