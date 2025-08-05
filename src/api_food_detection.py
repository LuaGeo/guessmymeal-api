from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import base64
import io
import json
import os

app = FastAPI(title="Food Detection API", version="1.0.0")

# Globals
model = None
model_path = "models/food_detection_model.pt"

@app.on_event("startup")
async def load_model():
    """Load YOLOv8 model at startup"""
    global model
    try:
        model = YOLO(model_path)
        print("✅ YOLO model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading YOLO model: {e}")

def annotate_image(result):
    """Get annotated image as base64 string"""
    annotated_pil = Image.fromarray(result.plot())
    buffer = io.BytesIO()
    annotated_pil.save(buffer, format='JPEG')
    encoded_img = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return encoded_img

@app.get("/")
async def root():
    return {"message": "Food Detection API is running."}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        # Read image from request
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Detect
        results = model.predict(image)
        result = results[0]

        # Annotated image
        annotated_img_base64 = annotate_image(result)

        # Detection details
        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                detections.append({
                    "class_name": model.names[int(box.cls.item())],
                    "confidence": float(box.conf.item()),
                    "bbox": [float(coord) for coord in box.xyxy[0].tolist()]
                })

        response = {
            "success": True,
            "detections": detections,
            "annotated_image": annotated_img_base64,
            "total_detections": len(detections)
        }

        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
